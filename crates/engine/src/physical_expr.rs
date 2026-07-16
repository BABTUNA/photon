//! Physical expressions (book: "Physical Plans").
//!
//! A physical expression is runnable: `evaluate(batch)` computes one output
//! column across every row of the input batch — vectorized, not row-at-a-
//! time. Unlike `LogicalExpr` (an enum the planners pattern-match on), the
//! physical side is a trait: nothing ever inspects a physical expression,
//! execution just calls it.

use std::cmp::Ordering;
use std::fmt::Display;
use std::sync::Arc;

use arrow::array::{BooleanArray, Float64Array, Int64Array};
use arrow::datatypes::DataType;

use crate::datatypes::{ArrowFieldVector, ColumnVector, LiteralValueVector, RecordBatch, ScalarValue};
use crate::logical_expr::{AggregateFunc, Operator};

pub trait Expression: Display + Send + Sync {
    /// Evaluate against `input`, producing a column with one value per row.
    fn evaluate(&self, input: &RecordBatch) -> Arc<dyn ColumnVector>;
}

/// Input column by POSITION. The logical layer references columns by name;
/// the query planner (2.8) resolves names against the input schema exactly
/// once, and execution is index-only from then on.
pub struct ColumnExpression {
    pub index: usize,
}

impl Expression for ColumnExpression {
    fn evaluate(&self, input: &RecordBatch) -> Arc<dyn ColumnVector> {
        // No work: hand out another reference to the batch's column.
        Arc::clone(input.column(self.index))
    }
}

impl Display for ColumnExpression {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "#{}", self.index)
    }
}

/// A constant. Evaluates to a `LiteralValueVector` sized to the input batch
/// — the 1.4 payoff: `price * 0.9` over 10k rows never materializes 10k
/// copies of `0.9`. One struct covers every type; `ScalarValue` already
/// unifies what KQuery needs three Literal*Expression classes for.
pub struct LiteralExpression {
    pub value: ScalarValue,
}

impl Expression for LiteralExpression {
    fn evaluate(&self, input: &RecordBatch) -> Arc<dyn ColumnVector> {
        Arc::new(LiteralValueVector::new(
            self.value.data_type(),
            Some(self.value.clone()),
            input.row_count(),
        ))
    }
}

impl Display for LiteralExpression {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.value)
    }
}

/// Two child expressions combined by an operator, evaluated per batch.
///
/// Coercion policy (simpler than SQL, honest for a learning engine):
/// - comparisons work across any numeric mix (values normalized to i64/f64),
///   plus string-vs-string and bool-vs-bool;
/// - math coerces to Float64 if either side is floating, else Int64;
/// - a NULL on either side makes that row's result NULL.
pub struct BinaryExpression {
    pub left: Arc<dyn Expression>,
    pub op: Operator,
    pub right: Arc<dyn Expression>,
}

impl Expression for BinaryExpression {
    fn evaluate(&self, input: &RecordBatch) -> Arc<dyn ColumnVector> {
        let l = self.left.evaluate(input);
        let r = self.right.evaluate(input);
        assert_eq!(l.size(), r.size(), "operand column lengths differ");
        let n = l.size();

        use Operator::*;
        match self.op {
            Eq | Neq | Gt | GtEq | Lt | LtEq => {
                let values = (0..n).map(|i| match (l.value(i), r.value(i)) {
                    (Some(a), Some(b)) => Some(compare(&a, &b, self.op)),
                    _ => None, // comparing against NULL yields NULL
                });
                wrap(BooleanArray::from_iter(values))
            }
            And | Or => {
                let values = (0..n).map(|i| match (l.value(i), r.value(i)) {
                    (Some(a), Some(b)) => Some(match self.op {
                        And => as_bool(&a) && as_bool(&b),
                        _ => as_bool(&a) || as_bool(&b),
                    }),
                    _ => None, // simplified: no SQL three-valued logic
                });
                wrap(BooleanArray::from_iter(values))
            }
            Add | Subtract | Multiply | Divide | Modulus => {
                // Widen once per batch, not per row.
                let float = is_float(l.data_type()) || is_float(r.data_type());
                if float {
                    let values = (0..n).map(|i| match (l.value(i), r.value(i)) {
                        (Some(a), Some(b)) => Some(float_math(as_f64(&a), as_f64(&b), self.op)),
                        _ => None,
                    });
                    wrap(Float64Array::from_iter(values))
                } else {
                    let values = (0..n).map(|i| match (l.value(i), r.value(i)) {
                        (Some(a), Some(b)) => Some(int_math(as_i64(&a), as_i64(&b), self.op)),
                        _ => None,
                    });
                    wrap(Int64Array::from_iter(values))
                }
            }
        }
    }
}

impl Display for BinaryExpression {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} {} {}", self.left, self.op, self.right)
    }
}

fn wrap(array: impl arrow::array::Array + 'static) -> Arc<dyn ColumnVector> {
    Arc::new(ArrowFieldVector::new(Arc::new(array)))
}

fn is_float(t: &DataType) -> bool {
    matches!(t, DataType::Float32 | DataType::Float64)
}

/// Normalize a numeric value for integer math. Panics on non-numeric input —
/// the logical layer should never have let that plan through.
fn as_i64(v: &ScalarValue) -> i64 {
    match v {
        ScalarValue::Int8(x) => *x as i64,
        ScalarValue::Int16(x) => *x as i64,
        ScalarValue::Int32(x) => *x as i64,
        ScalarValue::Int64(x) => *x,
        ScalarValue::UInt8(x) => *x as i64,
        ScalarValue::UInt16(x) => *x as i64,
        ScalarValue::UInt32(x) => *x as i64,
        ScalarValue::UInt64(x) => *x as i64,
        other => panic!("expected a numeric value, got {other}"),
    }
}

fn as_f64(v: &ScalarValue) -> f64 {
    match v {
        ScalarValue::Float32(x) => *x as f64,
        ScalarValue::Float64(x) => *x,
        other => as_i64(other) as f64,
    }
}

fn as_bool(v: &ScalarValue) -> bool {
    match v {
        ScalarValue::Boolean(b) => *b,
        other => panic!("AND/OR need boolean operands, got {other}"),
    }
}

/// Compare two (non-null) values after coercing to a common domain.
fn compare(a: &ScalarValue, b: &ScalarValue, op: Operator) -> bool {
    let ord = match (a, b) {
        (ScalarValue::Utf8(x), ScalarValue::Utf8(y)) => x.cmp(y),
        (ScalarValue::Boolean(x), ScalarValue::Boolean(y)) => x.cmp(y),
        _ if is_float(&a.data_type()) || is_float(&b.data_type()) => {
            as_f64(a).total_cmp(&as_f64(b))
        }
        _ => as_i64(a).cmp(&as_i64(b)),
    };
    match op {
        Operator::Eq => ord == Ordering::Equal,
        Operator::Neq => ord != Ordering::Equal,
        Operator::Gt => ord == Ordering::Greater,
        Operator::GtEq => ord != Ordering::Less,
        Operator::Lt => ord == Ordering::Less,
        Operator::LtEq => ord != Ordering::Greater,
        other => unreachable!("{other} is not a comparison"),
    }
}

fn int_math(a: i64, b: i64, op: Operator) -> i64 {
    match op {
        Operator::Add => a + b,
        Operator::Subtract => a - b,
        Operator::Multiply => a * b,
        Operator::Divide => a / b,
        Operator::Modulus => a % b,
        other => unreachable!("{other} is not math"),
    }
}

fn float_math(a: f64, b: f64, op: Operator) -> f64 {
    match op {
        Operator::Add => a + b,
        Operator::Subtract => a - b,
        Operator::Multiply => a * b,
        Operator::Divide => a / b,
        Operator::Modulus => a % b,
        other => unreachable!("{other} is not math"),
    }
}

/// Physical aggregate: which function, over which input expression.
/// Not an `Expression` — it doesn't map a batch to a column; it feeds
/// accumulators inside HashAggregateExec, one per group.
pub struct AggregateExpression {
    pub func: AggregateFunc,
    pub expr: Arc<dyn Expression>,
}

impl AggregateExpression {
    pub fn create_accumulator(&self) -> Box<dyn Accumulator> {
        match self.func {
            AggregateFunc::Sum => Box::new(SumAccumulator { sum: None }),
            AggregateFunc::Min => Box::new(MinMaxAccumulator {
                current: None,
                keep_if: Operator::Lt,
            }),
            AggregateFunc::Max => Box::new(MinMaxAccumulator {
                current: None,
                keep_if: Operator::Gt,
            }),
            AggregateFunc::Avg => Box::new(AvgAccumulator { sum: 0.0, count: 0 }),
            AggregateFunc::Count => Box::new(CountAccumulator { count: 0 }),
        }
    }
}

impl Display for AggregateExpression {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}({})", self.func.name(), self.expr)
    }
}

/// Running state for one aggregate within one group. SQL semantics: NULL
/// inputs are skipped, never accumulated.
pub trait Accumulator: Send {
    fn accumulate(&mut self, value: Option<ScalarValue>);
    fn final_value(&self) -> Option<ScalarValue>;
}

/// MIN and MAX are the same machine with the comparison flipped.
struct MinMaxAccumulator {
    current: Option<ScalarValue>,
    keep_if: Operator,
}

impl Accumulator for MinMaxAccumulator {
    fn accumulate(&mut self, value: Option<ScalarValue>) {
        let Some(v) = value else { return };
        match &self.current {
            None => self.current = Some(v),
            Some(c) => {
                if compare(&v, c, self.keep_if) {
                    self.current = Some(v);
                }
            }
        }
    }

    fn final_value(&self) -> Option<ScalarValue> {
        self.current.clone()
    }
}

/// Sums integers in i64 and floats in f64; promotes to float if a float
/// ever shows up. Empty (or all-NULL) input sums to NULL, per SQL.
struct SumAccumulator {
    sum: Option<ScalarValue>,
}

impl Accumulator for SumAccumulator {
    fn accumulate(&mut self, value: Option<ScalarValue>) {
        let Some(v) = value else { return };
        let float = is_float(&v.data_type());
        match &mut self.sum {
            None => {
                self.sum = Some(if float {
                    ScalarValue::Float64(as_f64(&v))
                } else {
                    ScalarValue::Int64(as_i64(&v))
                })
            }
            Some(ScalarValue::Int64(s)) => {
                if float {
                    self.sum = Some(ScalarValue::Float64(*s as f64 + as_f64(&v)));
                } else {
                    *s += as_i64(&v);
                }
            }
            Some(ScalarValue::Float64(s)) => *s += as_f64(&v),
            Some(other) => unreachable!("sum state is always Int64/Float64, got {other}"),
        }
    }

    fn final_value(&self) -> Option<ScalarValue> {
        self.sum.clone()
    }
}

struct AvgAccumulator {
    sum: f64,
    count: i64,
}

impl Accumulator for AvgAccumulator {
    fn accumulate(&mut self, value: Option<ScalarValue>) {
        let Some(v) = value else { return };
        self.sum += as_f64(&v);
        self.count += 1;
    }

    fn final_value(&self) -> Option<ScalarValue> {
        (self.count > 0).then(|| ScalarValue::Float64(self.sum / self.count as f64))
    }
}

/// COUNT(expr): number of non-NULL values.
struct CountAccumulator {
    count: i64,
}

impl Accumulator for CountAccumulator {
    fn accumulate(&mut self, value: Option<ScalarValue>) {
        if value.is_some() {
            self.count += 1;
        }
    }

    fn final_value(&self) -> Option<ScalarValue> {
        Some(ScalarValue::Int64(self.count))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Int64Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch as ArrowRecordBatch;

    fn test_batch() -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, false),
        ]));
        ArrowRecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(vec![1, 2, 3])),
                Arc::new(StringArray::from(vec!["a", "b", "c"])),
            ],
        )
        .unwrap()
        .into()
    }

    #[test]
    fn column_expression_returns_the_indexed_column() {
        let batch = test_batch();
        let expr = ColumnExpression { index: 1 };

        let col = expr.evaluate(&batch);
        assert_eq!(col.size(), 3);
        assert_eq!(col.value(0), Some(ScalarValue::Utf8("a".to_string())));
        assert_eq!(expr.to_string(), "#1");
    }

    #[test]
    fn literal_expression_sizes_itself_to_the_batch() {
        let batch = test_batch();
        let expr = LiteralExpression {
            value: ScalarValue::Float64(0.9),
        };

        let col = expr.evaluate(&batch);
        assert_eq!(col.size(), 3);
        assert_eq!(col.data_type(), &DataType::Float64);
        assert_eq!(col.value(2), Some(ScalarValue::Float64(0.9)));
        assert_eq!(expr.to_string(), "0.9");
    }

    #[test]
    fn expressions_compose_as_trait_objects() {
        let batch = test_batch();
        let exprs: Vec<Arc<dyn Expression>> = vec![
            Arc::new(ColumnExpression { index: 0 }),
            Arc::new(LiteralExpression {
                value: ScalarValue::Int64(7),
            }),
        ];

        let cols: Vec<_> = exprs.iter().map(|e| e.evaluate(&batch)).collect();
        assert_eq!(cols[0].value(1), Some(ScalarValue::Int64(2)));
        assert_eq!(cols[1].value(1), Some(ScalarValue::Int64(7)));
    }

    fn bin(left: Arc<dyn Expression>, op: Operator, right: Arc<dyn Expression>) -> BinaryExpression {
        BinaryExpression { left, op, right }
    }

    fn col_at(index: usize) -> Arc<dyn Expression> {
        Arc::new(ColumnExpression { index })
    }

    fn lit_val(value: ScalarValue) -> Arc<dyn Expression> {
        Arc::new(LiteralExpression { value })
    }

    /// id: Int64 [1, 2, null], score: Float64 [0.5, 2.5, 9.0], name: Utf8.
    fn null_batch() -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, true),
            Field::new("score", DataType::Float64, false),
            Field::new("name", DataType::Utf8, false),
        ]));
        ArrowRecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(vec![Some(1), Some(2), None])),
                Arc::new(Float64Array::from(vec![0.5, 2.5, 9.0])),
                Arc::new(StringArray::from(vec!["ann", "bob", "ann"])),
            ],
        )
        .unwrap()
        .into()
    }

    #[test]
    fn integer_comparison_yields_booleans() {
        let batch = null_batch();
        let expr = bin(col_at(0), Operator::Gt, lit_val(ScalarValue::Int64(1)));

        let out = expr.evaluate(&batch);
        assert_eq!(out.data_type(), &DataType::Boolean);
        assert_eq!(out.value(0), Some(ScalarValue::Boolean(false)));
        assert_eq!(out.value(1), Some(ScalarValue::Boolean(true)));
        assert_eq!(out.value(2), None); // NULL comparison → NULL
    }

    #[test]
    fn mixed_int_float_comparison_coerces() {
        let batch = null_batch();
        // Int64 column vs Float64 column: both normalized to f64.
        let expr = bin(col_at(0), Operator::Lt, col_at(1));

        let out = expr.evaluate(&batch);
        assert_eq!(out.value(0), Some(ScalarValue::Boolean(false))); // 1 < 0.5
        assert_eq!(out.value(1), Some(ScalarValue::Boolean(true))); // 2 < 2.5
    }

    #[test]
    fn string_equality_works() {
        let batch = null_batch();
        let expr = bin(col_at(2), Operator::Eq, lit_val(ScalarValue::Utf8("ann".into())));

        let out = expr.evaluate(&batch);
        assert_eq!(out.value(0), Some(ScalarValue::Boolean(true)));
        assert_eq!(out.value(1), Some(ScalarValue::Boolean(false)));
        assert_eq!(out.value(2), Some(ScalarValue::Boolean(true)));
    }

    #[test]
    fn integer_math_stays_integer() {
        let batch = null_batch();
        let expr = bin(col_at(0), Operator::Multiply, lit_val(ScalarValue::Int64(10)));

        let out = expr.evaluate(&batch);
        assert_eq!(out.data_type(), &DataType::Int64);
        assert_eq!(out.value(1), Some(ScalarValue::Int64(20)));
        assert_eq!(out.value(2), None); // NULL propagates through math
    }

    #[test]
    fn mixed_math_widens_to_float64() {
        let batch = null_batch();
        let expr = bin(col_at(0), Operator::Add, col_at(1));

        let out = expr.evaluate(&batch);
        assert_eq!(out.data_type(), &DataType::Float64);
        assert_eq!(out.value(0), Some(ScalarValue::Float64(1.5)));
    }

    #[test]
    fn boolean_algebra_composes() {
        let batch = null_batch();
        // (id >= 1) AND (score < 3.0)
        let expr = bin(
            Arc::new(bin(col_at(0), Operator::GtEq, lit_val(ScalarValue::Int64(1)))),
            Operator::And,
            Arc::new(bin(col_at(1), Operator::Lt, lit_val(ScalarValue::Float64(3.0)))),
        );

        let out = expr.evaluate(&batch);
        assert_eq!(out.value(0), Some(ScalarValue::Boolean(true)));
        assert_eq!(out.value(1), Some(ScalarValue::Boolean(true)));
        assert_eq!(out.value(2), None); // NULL AND false → NULL (simplified)
        assert_eq!(expr.to_string(), "#0 >= 1 AND #1 < 3");
    }

    #[test]
    #[should_panic(expected = "expected a numeric value")]
    fn math_on_strings_panics() {
        let batch = null_batch();
        let expr = bin(col_at(2), Operator::Add, lit_val(ScalarValue::Int64(1)));
        expr.evaluate(&batch);
    }
}
