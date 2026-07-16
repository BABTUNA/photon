//! Physical expressions (book: "Physical Plans").
//!
//! A physical expression is runnable: `evaluate(batch)` computes one output
//! column across every row of the input batch — vectorized, not row-at-a-
//! time. Unlike `LogicalExpr` (an enum the planners pattern-match on), the
//! physical side is a trait: nothing ever inspects a physical expression,
//! execution just calls it.

use std::fmt::Display;
use std::sync::Arc;

use crate::datatypes::{ColumnVector, LiteralValueVector, RecordBatch, ScalarValue};

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
}
