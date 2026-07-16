//! The execution context — the object users hold (book: "Query Execution").
//!
//! Turns file paths into DataFrames, and — since 2.9 — runs them: `collect`
//! plans the DataFrame's logical tree into a physical one and drains it.

use std::sync::Arc;

use crate::dataframe::DataFrame;
use crate::datasource::{CsvDataSource, ParquetDataSource};
use crate::datatypes::RecordBatch;
use crate::logical_plan::Scan;
use crate::query_planner::create_physical_plan;

/// Default rows per batch for file sources.
pub const DEFAULT_BATCH_SIZE: usize = 1024;

#[derive(Default)]
pub struct ExecutionContext;

impl ExecutionContext {
    pub fn new() -> Self {
        Self
    }

    /// A DataFrame scanning a CSV file (schema inferred).
    pub fn csv(&self, path: &str) -> DataFrame {
        let source = Arc::new(CsvDataSource::new(path, None, DEFAULT_BATCH_SIZE));
        DataFrame::new(Arc::new(Scan::new(path, source, vec![])))
    }

    /// A DataFrame scanning a parquet file (schema from the footer).
    pub fn parquet(&self, path: &str) -> DataFrame {
        let source = Arc::new(ParquetDataSource::new(path));
        DataFrame::new(Arc::new(Scan::new(path, source, vec![])))
    }

    /// Run the query: logical plan → physical plan → drain every batch.
    /// The whole engine, in two lines.
    pub fn collect(&self, df: &DataFrame) -> Vec<RecordBatch> {
        let physical = create_physical_plan(df.logical_plan().as_ref());
        physical.execute().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datatypes::ScalarValue;

    #[test]
    fn collect_runs_a_bare_scan() {
        let ctx = ExecutionContext::new();
        let df = ctx.csv("testdata/employee.csv");

        let batches = ctx.collect(&df);
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].row_count(), 4);
        assert_eq!(batches[0].column_count(), 6);
        assert_eq!(batches[0].column(0).value(0), Some(ScalarValue::Int64(1)));
    }
}
