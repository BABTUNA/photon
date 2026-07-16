//! Physical expressions (book: "Physical Plans").
//!
//! A physical expression is runnable: `evaluate(batch)` computes one output
//! column across every row of the input batch — vectorized, not row-at-a-
//! time. Unlike `LogicalExpr` (an enum the planners pattern-match on), the
//! physical side is a trait: nothing ever inspects a physical expression,
//! execution just calls it.

use std::fmt::Display;
use std::sync::Arc;

use crate::datatypes::{ColumnVector, RecordBatch};

pub trait Expression: Display + Send + Sync {
    /// Evaluate against `input`, producing a column with one value per row.
    fn evaluate(&self, input: &RecordBatch) -> Arc<dyn ColumnVector>;
}
