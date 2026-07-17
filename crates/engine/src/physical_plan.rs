//! Physical plans (book: "Physical Plans").
//!
//! This is the other side of the what/how seam. A logical plan says WHAT
//! (`Selection: #state = 'CO'`); a physical plan is runnable code with all
//! decisions made — which columns by INDEX, which algorithm, which access
//! path. The QueryPlanner (2.8) translates one into the other; keeping the
//! two apart is what gives the optimizer (W3) a place to stand.

use std::collections::HashMap;
use std::fmt::Display;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use arrow::array::{
    BooleanArray, Float32Array, Float64Array, Int8Array, Int16Array, Int32Array, Int64Array,
    StringArray, UInt8Array, UInt16Array, UInt32Array, UInt64Array,
};
use arrow::datatypes::{DataType, SchemaRef};

use crate::datasource::DataSource;
use crate::datatypes::{ArrowFieldVector, ColumnVector, RecordBatch, ScalarValue, project_by_name};
use crate::logical_plan::JoinType;
use crate::physical_expr::{AggregateExpression, Expression};

pub trait PhysicalPlan: Display + Send + Sync {
    /// Schema of the batches `execute` yields.
    fn schema(&self) -> SchemaRef;

    /// Run this operator. Pull-based (volcano-style, but vectorized): the
    /// consumer drains the iterator, and each pull ripples down through the
    /// children — one RecordBatch at a time, never the whole table.
    fn execute(&self) -> Box<dyn Iterator<Item = RecordBatch>>;

    fn children(&self) -> Vec<Arc<dyn PhysicalPlan>>;
}

/// Leaf executor: delegate straight to the data source. All the real work
/// (projection pushdown included) happened in W1's source layer.
pub struct ScanExec {
    pub data_source: Arc<dyn DataSource>,
    pub projection: Vec<String>,
    schema: SchemaRef,
}

impl ScanExec {
    pub fn new(data_source: Arc<dyn DataSource>, projection: Vec<String>) -> Self {
        let schema = if projection.is_empty() {
            data_source.schema()
        } else {
            let names: Vec<&str> = projection.iter().map(String::as_str).collect();
            Arc::new(project_by_name(&data_source.schema(), &names).unwrap())
        };
        Self {
            data_source,
            projection,
            schema,
        }
    }
}

impl Display for ScanExec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ScanExec: projection={:?}", self.projection)
    }
}

impl PhysicalPlan for ScanExec {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn execute(&self) -> Box<dyn Iterator<Item = RecordBatch>> {
        self.data_source.scan(&self.projection)
    }

    fn children(&self) -> Vec<Arc<dyn PhysicalPlan>> {
        vec![]
    }
}

/// Evaluate one expression per output column against each input batch.
pub struct ProjectionExec {
    pub input: Arc<dyn PhysicalPlan>,
    pub exprs: Vec<Arc<dyn Expression>>,
    /// Provided by the planner (2.8), which derived it from the logical plan.
    schema: SchemaRef,
}

impl ProjectionExec {
    pub fn new(
        input: Arc<dyn PhysicalPlan>,
        schema: SchemaRef,
        exprs: Vec<Arc<dyn Expression>>,
    ) -> Self {
        Self {
            input,
            exprs,
            schema,
        }
    }
}

impl Display for ProjectionExec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let exprs: Vec<String> = self.exprs.iter().map(|e| e.to_string()).collect();
        write!(f, "ProjectionExec: {}", exprs.join(", "))
    }
}

impl PhysicalPlan for ProjectionExec {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn execute(&self) -> Box<dyn Iterator<Item = RecordBatch>> {
        let schema = Arc::clone(&self.schema);
        let exprs = self.exprs.clone();
        Box::new(self.input.execute().map(move |batch| {
            let columns = exprs.iter().map(|e| e.evaluate(&batch)).collect();
            RecordBatch::new(Arc::clone(&schema), columns)
        }))
    }

    fn children(&self) -> Vec<Arc<dyn PhysicalPlan>> {
        vec![Arc::clone(&self.input)]
    }
}

/// Keep rows where the predicate evaluates to true (NULL filters out, per
/// SQL WHERE semantics). Schema passes through unchanged.
pub struct SelectionExec {
    pub input: Arc<dyn PhysicalPlan>,
    pub expr: Arc<dyn Expression>,
}

impl SelectionExec {
    pub fn new(input: Arc<dyn PhysicalPlan>, expr: Arc<dyn Expression>) -> Self {
        Self { input, expr }
    }
}

impl Display for SelectionExec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SelectionExec: {}", self.expr)
    }
}

impl PhysicalPlan for SelectionExec {
    fn schema(&self) -> SchemaRef {
        self.input.schema()
    }

    fn execute(&self) -> Box<dyn Iterator<Item = RecordBatch>> {
        let expr = Arc::clone(&self.expr);
        let schema = self.schema();
        Box::new(self.input.execute().map(move |batch| {
            let predicate = expr.evaluate(&batch);
            let keep: Vec<bool> = (0..batch.row_count())
                .map(|i| matches!(predicate.value(i), Some(ScalarValue::Boolean(true))))
                .collect();
            let columns = (0..batch.column_count())
                .map(|c| filter_column(batch.column(c), &keep))
                .collect();
            RecordBatch::new(Arc::clone(&schema), columns)
        }))
    }

    fn children(&self) -> Vec<Arc<dyn PhysicalPlan>> {
        vec![Arc::clone(&self.input)]
    }
}

/// Gather the rows of `col` where `keep` is true into a fresh column.
fn filter_column(col: &Arc<dyn ColumnVector>, keep: &[bool]) -> Arc<dyn ColumnVector> {
    let values = (0..col.size())
        .filter(|&i| keep[i])
        .map(|i| col.value(i))
        .collect();
    build_column(values, col.data_type())
}

/// Build a concrete arrow-backed column of `data_type` from scalar values.
fn build_column(values: Vec<Option<ScalarValue>>, data_type: &DataType) -> Arc<dyn ColumnVector> {
    macro_rules! build {
        ($variant:ident, $array:ty) => {{
            let iter = values.into_iter().map(|v| {
                v.map(|v| match v {
                    ScalarValue::$variant(x) => x,
                    other => panic!("value {other} does not fit column type {data_type:?}"),
                })
            });
            Arc::new(ArrowFieldVector::new(Arc::new(<$array>::from_iter(iter))))
                as Arc<dyn ColumnVector>
        }};
    }
    match data_type {
        DataType::Boolean => build!(Boolean, BooleanArray),
        DataType::Int8 => build!(Int8, Int8Array),
        DataType::Int16 => build!(Int16, Int16Array),
        DataType::Int32 => build!(Int32, Int32Array),
        DataType::Int64 => build!(Int64, Int64Array),
        DataType::UInt8 => build!(UInt8, UInt8Array),
        DataType::UInt16 => build!(UInt16, UInt16Array),
        DataType::UInt32 => build!(UInt32, UInt32Array),
        DataType::UInt64 => build!(UInt64, UInt64Array),
        DataType::Float32 => build!(Float32, Float32Array),
        DataType::Float64 => build!(Float64, Float64Array),
        DataType::Utf8 => build!(Utf8, StringArray),
        other => panic!("unsupported column type {other:?}"),
    }
}

/// Group rows by key expressions, feed each group's rows through per-group
/// accumulators, emit one row per group. A PIPELINE BREAKER: unlike
/// Projection/Selection it cannot stream — every input batch must be seen
/// before any group is final.
pub struct HashAggregateExec {
    pub input: Arc<dyn PhysicalPlan>,
    pub group_exprs: Vec<Arc<dyn Expression>>,
    pub aggregate_exprs: Vec<AggregateExpression>,
    /// Group fields then aggregate fields — from the planner (2.8).
    schema: SchemaRef,
}

impl HashAggregateExec {
    pub fn new(
        input: Arc<dyn PhysicalPlan>,
        schema: SchemaRef,
        group_exprs: Vec<Arc<dyn Expression>>,
        aggregate_exprs: Vec<AggregateExpression>,
    ) -> Self {
        Self {
            input,
            group_exprs,
            aggregate_exprs,
            schema,
        }
    }
}

impl Display for HashAggregateExec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let groups: Vec<String> = self.group_exprs.iter().map(|e| e.to_string()).collect();
        let aggs: Vec<String> = self.aggregate_exprs.iter().map(|a| a.to_string()).collect();
        write!(
            f,
            "HashAggregateExec: groupExpr=[{}], aggregateExpr=[{}]",
            groups.join(", "),
            aggs.join(", ")
        )
    }
}

impl PhysicalPlan for HashAggregateExec {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn execute(&self) -> Box<dyn Iterator<Item = RecordBatch>> {
        let mut groups: HashMap<GroupKey, Vec<Box<dyn crate::physical_expr::Accumulator>>> =
            HashMap::new();

        for batch in self.input.execute() {
            // Evaluate group keys and aggregate inputs once per batch...
            let key_cols: Vec<_> = self.group_exprs.iter().map(|e| e.evaluate(&batch)).collect();
            let input_cols: Vec<_> = self
                .aggregate_exprs
                .iter()
                .map(|a| a.expr.evaluate(&batch))
                .collect();
            // ...then route each row to its group's accumulators.
            for row in 0..batch.row_count() {
                let key = GroupKey(key_cols.iter().map(|c| c.value(row)).collect());
                let accumulators = groups.entry(key).or_insert_with(|| {
                    self.aggregate_exprs
                        .iter()
                        .map(|a| a.create_accumulator())
                        .collect()
                });
                for (accumulator, col) in accumulators.iter_mut().zip(&input_cols) {
                    accumulator.accumulate(col.value(row));
                }
            }
        }

        // One output row per group: key values, then accumulator results.
        let n_cols = self.schema.fields().len();
        let mut out: Vec<Vec<Option<ScalarValue>>> = vec![Vec::new(); n_cols];
        for (key, accumulators) in groups {
            for (i, key_value) in key.0.into_iter().enumerate() {
                out[i].push(key_value);
            }
            for (i, accumulator) in accumulators.iter().enumerate() {
                out[self.group_exprs.len() + i].push(accumulator.final_value());
            }
        }
        let columns: Vec<_> = out
            .into_iter()
            .zip(self.schema.fields())
            .map(|(values, field)| build_column(values, field.data_type()))
            .collect();
        Box::new(std::iter::once(RecordBatch::new(
            Arc::clone(&self.schema),
            columns,
        )))
    }

    fn children(&self) -> Vec<Arc<dyn PhysicalPlan>> {
        vec![Arc::clone(&self.input)]
    }
}

/// Classic two-phase hash join.
///
/// BUILD: drain the LEFT input into a hash table keyed on the join keys.
/// PROBE: stream the RIGHT input, looking each row up in the table.
/// Inner emits matches only; Left additionally emits unmatched build rows
/// padded with NULLs; Right emits unmatched probe rows padded with NULLs.
///
/// SQL subtlety honored here: a NULL join key matches NOTHING — not even
/// another NULL (unlike GROUP BY, where NULL keys group together).
pub struct HashJoinExec {
    pub left: Arc<dyn PhysicalPlan>,
    pub right: Arc<dyn PhysicalPlan>,
    pub join_type: JoinType,
    pub left_keys: Vec<usize>,
    pub right_keys: Vec<usize>,
    schema: SchemaRef,
}

impl HashJoinExec {
    pub fn new(
        left: Arc<dyn PhysicalPlan>,
        right: Arc<dyn PhysicalPlan>,
        join_type: JoinType,
        left_keys: Vec<usize>,
        right_keys: Vec<usize>,
        schema: SchemaRef,
    ) -> Self {
        Self {
            left,
            right,
            join_type,
            left_keys,
            right_keys,
            schema,
        }
    }
}

impl Display for HashJoinExec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let on: Vec<String> = self
            .left_keys
            .iter()
            .zip(&self.right_keys)
            .map(|(l, r)| format!("#{l} = #{r}"))
            .collect();
        write!(
            f,
            "HashJoinExec: type={:?}, on=[{}]",
            self.join_type,
            on.join(", ")
        )
    }
}

impl PhysicalPlan for HashJoinExec {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn execute(&self) -> Box<dyn Iterator<Item = RecordBatch>> {
        // BUILD: materialize every left row; index the non-NULL-keyed ones.
        let mut left_rows: Vec<Vec<Option<ScalarValue>>> = Vec::new();
        let mut table: HashMap<GroupKey, Vec<usize>> = HashMap::new();
        for batch in self.left.execute() {
            for row in 0..batch.row_count() {
                let values: Vec<_> = (0..batch.column_count())
                    .map(|c| batch.column(c).value(row))
                    .collect();
                let key = GroupKey(self.left_keys.iter().map(|&k| values[k].clone()).collect());
                if !key.has_null() {
                    table.entry(key).or_default().push(left_rows.len());
                }
                left_rows.push(values);
            }
        }

        let left_width = self.left.schema().fields().len();
        let right_width = self.right.schema().fields().len();
        let mut left_matched = vec![false; left_rows.len()];
        let mut out_rows: Vec<Vec<Option<ScalarValue>>> = Vec::new();

        // PROBE: stream the right side through the table.
        for batch in self.right.execute() {
            for row in 0..batch.row_count() {
                let values: Vec<_> = (0..batch.column_count())
                    .map(|c| batch.column(c).value(row))
                    .collect();
                let key = GroupKey(self.right_keys.iter().map(|&k| values[k].clone()).collect());
                let matches = (!key.has_null()).then(|| table.get(&key)).flatten();
                match matches {
                    Some(indices) => {
                        for &i in indices {
                            left_matched[i] = true;
                            let mut joined = left_rows[i].clone();
                            joined.extend(values.iter().cloned());
                            out_rows.push(joined);
                        }
                    }
                    None => {
                        if self.join_type == JoinType::Right {
                            let mut joined = vec![None; left_width];
                            joined.extend(values.iter().cloned());
                            out_rows.push(joined);
                        }
                    }
                }
            }
        }

        // LEFT join: emit build rows nothing probed, padded on the right.
        if self.join_type == JoinType::Left {
            for (i, matched) in left_matched.iter().enumerate() {
                if !matched {
                    let mut joined = left_rows[i].clone();
                    joined.extend(std::iter::repeat_n(None, right_width));
                    out_rows.push(joined);
                }
            }
        }

        // Rows → columns → one output batch.
        let n_cols = left_width + right_width;
        let mut cols: Vec<Vec<Option<ScalarValue>>> = vec![Vec::with_capacity(out_rows.len()); n_cols];
        for row in out_rows {
            for (i, value) in row.into_iter().enumerate() {
                cols[i].push(value);
            }
        }
        let columns: Vec<_> = cols
            .into_iter()
            .zip(self.schema.fields())
            .map(|(values, field)| build_column(values, field.data_type()))
            .collect();
        Box::new(std::iter::once(RecordBatch::new(
            Arc::clone(&self.schema),
            columns,
        )))
    }

    fn children(&self) -> Vec<Arc<dyn PhysicalPlan>> {
        vec![Arc::clone(&self.left), Arc::clone(&self.right)]
    }
}

/// Hash-map key over group values. `ScalarValue` is only `PartialEq`
/// (floats: NaN != NaN), so it can't be a key directly — this wrapper
/// compares and hashes floats BY BITS: NaN groups with NaN, and 0.0 / -0.0
/// land in different groups. Both fine for a learning engine; noted.
struct GroupKey(Vec<Option<ScalarValue>>);

impl GroupKey {
    /// Joins need this: SQL says a NULL key equals nothing.
    fn has_null(&self) -> bool {
        self.0.iter().any(Option::is_none)
    }
}

fn scalar_bits_eq(a: &ScalarValue, b: &ScalarValue) -> bool {
    match (a, b) {
        (ScalarValue::Float32(x), ScalarValue::Float32(y)) => x.to_bits() == y.to_bits(),
        (ScalarValue::Float64(x), ScalarValue::Float64(y)) => x.to_bits() == y.to_bits(),
        _ => a == b,
    }
}

impl PartialEq for GroupKey {
    fn eq(&self, other: &Self) -> bool {
        self.0.len() == other.0.len()
            && self.0.iter().zip(&other.0).all(|(a, b)| match (a, b) {
                (None, None) => true,
                (Some(a), Some(b)) => scalar_bits_eq(a, b),
                _ => false,
            })
    }
}

impl Eq for GroupKey {}

impl Hash for GroupKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for value in &self.0 {
            match value {
                None => 0u8.hash(state),
                Some(v) => {
                    1u8.hash(state);
                    std::mem::discriminant(v).hash(state);
                    match v {
                        ScalarValue::Boolean(x) => x.hash(state),
                        ScalarValue::Int8(x) => x.hash(state),
                        ScalarValue::Int16(x) => x.hash(state),
                        ScalarValue::Int32(x) => x.hash(state),
                        ScalarValue::Int64(x) => x.hash(state),
                        ScalarValue::UInt8(x) => x.hash(state),
                        ScalarValue::UInt16(x) => x.hash(state),
                        ScalarValue::UInt32(x) => x.hash(state),
                        ScalarValue::UInt64(x) => x.hash(state),
                        ScalarValue::Float32(x) => x.to_bits().hash(state),
                        ScalarValue::Float64(x) => x.to_bits().hash(state),
                        ScalarValue::Utf8(x) => x.hash(state),
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datasource::InMemoryDataSource;
    use crate::logical_expr::Operator;
    use crate::physical_expr::{BinaryExpression, ColumnExpression, LiteralExpression};
    use arrow::datatypes::{Field, Schema};
    use arrow::record_batch::RecordBatch as ArrowRecordBatch;

    /// Two batches so operators are exercised as streams, not one-shots.
    fn source() -> Arc<dyn DataSource> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("state", DataType::Utf8, false),
            Field::new("salary", DataType::Int64, false),
        ]));
        let batch = |ids: Vec<i64>, states: Vec<&str>, salaries: Vec<i64>| {
            ArrowRecordBatch::try_new(
                Arc::clone(&schema),
                vec![
                    Arc::new(Int64Array::from(ids)),
                    Arc::new(StringArray::from(states)),
                    Arc::new(Int64Array::from(salaries)),
                ],
            )
            .unwrap()
            .into()
        };
        Arc::new(InMemoryDataSource::new(
            Arc::clone(&schema),
            vec![
                batch(vec![1, 2], vec!["CO", "CA"], vec![12000, 10000]),
                batch(vec![3], vec!["CO"], vec![11500]),
            ],
        ))
    }

    #[test]
    fn scan_exec_streams_source_batches() {
        let scan = ScanExec::new(source(), vec![]);

        assert_eq!(scan.schema().fields().len(), 3);
        let batches: Vec<_> = scan.execute().collect();
        assert_eq!(batches.len(), 2);
        assert_eq!(batches.iter().map(|b| b.row_count()).sum::<usize>(), 3);
        assert!(scan.children().is_empty());
    }

    #[test]
    fn projection_exec_computes_expression_columns() {
        let scan: Arc<dyn PhysicalPlan> = Arc::new(ScanExec::new(source(), vec![]));
        let out_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("mult", DataType::Int64, true),
        ]));
        let proj = ProjectionExec::new(
            scan,
            out_schema,
            vec![
                Arc::new(ColumnExpression { index: 0 }),
                Arc::new(BinaryExpression {
                    left: Arc::new(ColumnExpression { index: 2 }),
                    op: Operator::Multiply,
                    right: Arc::new(LiteralExpression {
                        value: ScalarValue::Int64(2),
                    }),
                }),
            ],
        );

        let batches: Vec<_> = proj.execute().collect();
        assert_eq!(batches[0].column_count(), 2);
        assert_eq!(batches[0].column(1).value(0), Some(ScalarValue::Int64(24000)));
        assert_eq!(batches[1].column(1).value(0), Some(ScalarValue::Int64(23000)));
        assert_eq!(proj.to_string(), "ProjectionExec: #0, #2 * 2");
    }

    #[test]
    fn selection_exec_filters_within_each_batch() {
        let scan: Arc<dyn PhysicalPlan> = Arc::new(ScanExec::new(source(), vec![]));
        let sel = SelectionExec::new(
            scan,
            Arc::new(BinaryExpression {
                left: Arc::new(ColumnExpression { index: 1 }),
                op: Operator::Eq,
                right: Arc::new(LiteralExpression {
                    value: ScalarValue::Utf8("CO".to_string()),
                }),
            }),
        );

        let batches: Vec<_> = sel.execute().collect();
        // Schema unchanged, rows filtered per batch: [1 of 2], [1 of 1].
        assert_eq!(sel.schema().fields().len(), 3);
        assert_eq!(batches[0].row_count(), 1);
        assert_eq!(batches[0].column(0).value(0), Some(ScalarValue::Int64(1)));
        assert_eq!(batches[1].row_count(), 1);
        assert_eq!(batches[1].column(0).value(0), Some(ScalarValue::Int64(3)));
        assert_eq!(sel.to_string(), "SelectionExec: #1 = 'CO'");
    }

    /// Find the output row whose first (group-key) column equals `key`.
    fn group_row(batch: &RecordBatch, key: &str) -> usize {
        (0..batch.row_count())
            .find(|&i| batch.column(0).value(i) == Some(ScalarValue::Utf8(key.to_string())))
            .unwrap_or_else(|| panic!("no group {key:?} in output"))
    }

    #[test]
    fn hash_aggregate_groups_across_batches() {
        use crate::logical_expr::AggregateFunc;

        let scan: Arc<dyn PhysicalPlan> = Arc::new(ScanExec::new(source(), vec![]));
        let out_schema = Arc::new(Schema::new(vec![
            Field::new("state", DataType::Utf8, false),
            Field::new("SUM", DataType::Int64, true),
            Field::new("COUNT", DataType::Int64, true),
        ]));
        let agg = HashAggregateExec::new(
            scan,
            out_schema,
            vec![Arc::new(ColumnExpression { index: 1 })],
            vec![
                AggregateExpression {
                    func: AggregateFunc::Sum,
                    expr: Arc::new(ColumnExpression { index: 2 }),
                },
                AggregateExpression {
                    func: AggregateFunc::Count,
                    expr: Arc::new(ColumnExpression { index: 0 }),
                },
            ],
        );

        let batches: Vec<_> = agg.execute().collect();
        // Pipeline breaker: two input batches, ONE output batch.
        assert_eq!(batches.len(), 1);
        let batch = &batches[0];
        assert_eq!(batch.row_count(), 2); // CO, CA

        // CO spans both input batches: 12000 + 11500.
        let co = group_row(batch, "CO");
        assert_eq!(batch.column(1).value(co), Some(ScalarValue::Int64(23500)));
        assert_eq!(batch.column(2).value(co), Some(ScalarValue::Int64(2)));

        let ca = group_row(batch, "CA");
        assert_eq!(batch.column(1).value(ca), Some(ScalarValue::Int64(10000)));
        assert_eq!(batch.column(2).value(ca), Some(ScalarValue::Int64(1)));
    }

    #[test]
    fn hash_aggregate_with_no_groups_is_one_global_row() {
        use crate::logical_expr::AggregateFunc;

        let scan: Arc<dyn PhysicalPlan> = Arc::new(ScanExec::new(source(), vec![]));
        let out_schema = Arc::new(Schema::new(vec![
            Field::new("MIN", DataType::Int64, true),
            Field::new("MAX", DataType::Int64, true),
            Field::new("AVG", DataType::Float64, true),
        ]));
        let salary = || -> Arc<dyn Expression> { Arc::new(ColumnExpression { index: 2 }) };
        let agg = HashAggregateExec::new(
            scan,
            out_schema,
            vec![], // no GROUP BY: every row keys to the empty tuple
            vec![
                AggregateExpression { func: AggregateFunc::Min, expr: salary() },
                AggregateExpression { func: AggregateFunc::Max, expr: salary() },
                AggregateExpression { func: AggregateFunc::Avg, expr: salary() },
            ],
        );

        let batch = agg.execute().next().unwrap();
        assert_eq!(batch.row_count(), 1);
        assert_eq!(batch.column(0).value(0), Some(ScalarValue::Int64(10000)));
        assert_eq!(batch.column(1).value(0), Some(ScalarValue::Int64(12000)));
        assert_eq!(batch.column(2).value(0), Some(ScalarValue::Float64(11166.666666666666)));
    }

    /// left: (key, l_val) — key 1 twice, key 2 once, one NULL key.
    /// right: (key, r_val) — key 1 once, key 3 once, one NULL key.
    fn join_sources() -> (Arc<dyn PhysicalPlan>, Arc<dyn PhysicalPlan>) {
        let make = |names: [&str; 2], keys: Vec<Option<i64>>, vals: Vec<&str>| {
            let schema = Arc::new(Schema::new(vec![
                Field::new(names[0], DataType::Int64, true),
                Field::new(names[1], DataType::Utf8, false),
            ]));
            let batch: RecordBatch = ArrowRecordBatch::try_new(
                Arc::clone(&schema),
                vec![
                    Arc::new(Int64Array::from(keys)),
                    Arc::new(StringArray::from(vals)),
                ],
            )
            .unwrap()
            .into();
            let source = Arc::new(InMemoryDataSource::new(schema, vec![batch]));
            Arc::new(ScanExec::new(source, vec![])) as Arc<dyn PhysicalPlan>
        };
        (
            make(["k", "l"], vec![Some(1), Some(1), Some(2), None], vec!["a", "b", "c", "d"]),
            make(["k2", "r"], vec![Some(1), Some(3), None], vec!["x", "y", "z"]),
        )
    }

    fn join_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int64, true),
            Field::new("l", DataType::Utf8, true),
            Field::new("k2", DataType::Int64, true),
            Field::new("r", DataType::Utf8, true),
        ]))
    }

    fn rows_of(batch: &RecordBatch) -> Vec<Vec<Option<ScalarValue>>> {
        let mut rows: Vec<Vec<_>> = (0..batch.row_count())
            .map(|i| (0..batch.column_count()).map(|c| batch.column(c).value(i)).collect())
            .collect();
        rows.sort_by_key(|r| format!("{r:?}"));
        rows
    }

    fn s(v: &str) -> Option<ScalarValue> {
        Some(ScalarValue::Utf8(v.to_string()))
    }

    fn i(v: i64) -> Option<ScalarValue> {
        Some(ScalarValue::Int64(v))
    }

    #[test]
    fn inner_join_matches_and_ignores_null_keys() {
        let (left, right) = join_sources();
        let join = HashJoinExec::new(left, right, JoinType::Inner, vec![0], vec![0], join_schema());

        let batch = join.execute().next().unwrap();
        // key 1: two left rows × one right row = 2 output rows. NULL keys never match.
        assert_eq!(
            rows_of(&batch),
            vec![
                vec![i(1), s("a"), i(1), s("x")],
                vec![i(1), s("b"), i(1), s("x")],
            ]
        );
        assert_eq!(join.to_string(), "HashJoinExec: type=Inner, on=[#0 = #0]");
    }

    #[test]
    fn left_join_pads_unmatched_build_rows() {
        let (left, right) = join_sources();
        let join = HashJoinExec::new(left, right, JoinType::Left, vec![0], vec![0], join_schema());

        let batch = join.execute().next().unwrap();
        // 2 matches + unmatched left rows: key 2, and the NULL-key row.
        // (rows_of sorts by debug string, so None-leading rows come first)
        assert_eq!(
            rows_of(&batch),
            vec![
                vec![None, s("d"), None, None],
                vec![i(1), s("a"), i(1), s("x")],
                vec![i(1), s("b"), i(1), s("x")],
                vec![i(2), s("c"), None, None],
            ]
        );
    }

    #[test]
    fn right_join_pads_unmatched_probe_rows() {
        let (left, right) = join_sources();
        let join = HashJoinExec::new(left, right, JoinType::Right, vec![0], vec![0], join_schema());

        let batch = join.execute().next().unwrap();
        assert_eq!(
            rows_of(&batch),
            vec![
                vec![None, None, None, s("z")],
                vec![None, None, i(3), s("y")],
                vec![i(1), s("a"), i(1), s("x")],
                vec![i(1), s("b"), i(1), s("x")],
            ]
        );
    }

    #[test]
    fn accumulators_skip_nulls() {
        use crate::logical_expr::AggregateFunc;

        let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, true)]));
        let data: RecordBatch = ArrowRecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Int64Array::from(vec![Some(10), None, Some(20)]))],
        )
        .unwrap()
        .into();
        let source = Arc::new(InMemoryDataSource::new(schema, vec![data]));

        let scan: Arc<dyn PhysicalPlan> = Arc::new(ScanExec::new(source, vec![]));
        let out_schema = Arc::new(Schema::new(vec![
            Field::new("COUNT", DataType::Int64, true),
            Field::new("AVG", DataType::Float64, true),
        ]));
        let v = || -> Arc<dyn Expression> { Arc::new(ColumnExpression { index: 0 }) };
        let agg = HashAggregateExec::new(
            scan,
            out_schema,
            vec![],
            vec![
                AggregateExpression { func: AggregateFunc::Count, expr: v() },
                AggregateExpression { func: AggregateFunc::Avg, expr: v() },
            ],
        );

        let batch = agg.execute().next().unwrap();
        assert_eq!(batch.column(0).value(0), Some(ScalarValue::Int64(2))); // NULL not counted
        assert_eq!(batch.column(1).value(0), Some(ScalarValue::Float64(15.0))); // (10+20)/2
    }
}
