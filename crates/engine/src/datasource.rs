//! The data source contract (book: "Data Sources").
//!
//! Everything the engine reads — in-memory tables, CSV, parquet, and later
//! Part 2's multimodal clip table — implements one small trait: describe
//! your schema, and scan yourself into a stream of record batches.

use arrow::datatypes::SchemaRef;

use crate::datatypes::RecordBatch;

pub trait DataSource {
    /// The full schema of the underlying data, before any projection.
    fn schema(&self) -> SchemaRef;

    /// Read the data, yielding batches containing only the `projection`
    /// columns, in that order. An empty projection means all columns.
    ///
    /// Projection lives here, at the lowest layer, because the cheapest
    /// byte is the one never read: a columnar file can skip whole column
    /// chunks on disk (1.9) instead of dropping them after the fact.
    fn scan(&self, projection: &[String]) -> Box<dyn Iterator<Item = RecordBatch>>;
}
