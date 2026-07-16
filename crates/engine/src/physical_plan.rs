//! Physical plans (book: "Physical Plans").
//!
//! This is the other side of the what/how seam. A logical plan says WHAT
//! (`Selection: #state = 'CO'`); a physical plan is runnable code with all
//! decisions made — which columns by INDEX, which algorithm, which access
//! path. The QueryPlanner (2.8) translates one into the other; keeping the
//! two apart is what gives the optimizer (W3) a place to stand.

use std::fmt::Display;
use std::sync::Arc;

use arrow::datatypes::SchemaRef;

use crate::datatypes::RecordBatch;

pub trait PhysicalPlan: Display + Send + Sync {
    /// Schema of the batches `execute` yields.
    fn schema(&self) -> SchemaRef;

    /// Run this operator. Pull-based (volcano-style, but vectorized): the
    /// consumer drains the iterator, and each pull ripples down through the
    /// children — one RecordBatch at a time, never the whole table.
    fn execute(&self) -> Box<dyn Iterator<Item = RecordBatch>>;

    fn children(&self) -> Vec<Arc<dyn PhysicalPlan>>;
}
