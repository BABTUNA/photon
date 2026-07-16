//! Physical plans (book: "Physical Plans").
//!
//! This is the other side of the what/how seam. A logical plan says WHAT
//! (`Selection: #state = 'CO'`); a physical plan is runnable code with all
//! decisions made — which columns by INDEX, which algorithm, which access
//! path. The QueryPlanner (2.8) translates one into the other; keeping the
//! two apart is what gives the optimizer (W3) a place to stand.

use std::fmt::Display;
use std::sync::Arc;

use arrow::array::{
    BooleanArray, Float32Array, Float64Array, Int8Array, Int16Array, Int32Array, Int64Array,
    StringArray, UInt8Array, UInt16Array, UInt32Array, UInt64Array,
};
use arrow::datatypes::{DataType, SchemaRef};

use crate::datasource::DataSource;
use crate::datatypes::{ArrowFieldVector, ColumnVector, RecordBatch, ScalarValue, project_by_name};
use crate::physical_expr::Expression;

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
    macro_rules! gather {
        ($variant:ident, $array:ty) => {{
            let values = (0..col.size()).filter(|&i| keep[i]).map(|i| {
                col.value(i).map(|v| match v {
                    ScalarValue::$variant(x) => x,
                    other => panic!("column changed type mid-batch: {other}"),
                })
            });
            Arc::new(ArrowFieldVector::new(Arc::new(<$array>::from_iter(values))))
                as Arc<dyn ColumnVector>
        }};
    }
    match col.data_type() {
        DataType::Boolean => gather!(Boolean, BooleanArray),
        DataType::Int8 => gather!(Int8, Int8Array),
        DataType::Int16 => gather!(Int16, Int16Array),
        DataType::Int32 => gather!(Int32, Int32Array),
        DataType::Int64 => gather!(Int64, Int64Array),
        DataType::UInt8 => gather!(UInt8, UInt8Array),
        DataType::UInt16 => gather!(UInt16, UInt16Array),
        DataType::UInt32 => gather!(UInt32, UInt32Array),
        DataType::UInt64 => gather!(UInt64, UInt64Array),
        DataType::Float32 => gather!(Float32, Float32Array),
        DataType::Float64 => gather!(Float64, Float64Array),
        DataType::Utf8 => gather!(Utf8, StringArray),
        other => panic!("SelectionExec: unsupported column type {other:?}"),
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
}
