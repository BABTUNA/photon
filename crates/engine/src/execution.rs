//! The execution context — the object users hold (book: "DataFrames").
//!
//! For now it only turns file paths into DataFrames; `collect()` — actually
//! pulling results through a physical plan — lands here in 2.9.

use std::sync::Arc;

use crate::dataframe::DataFrame;
use crate::datasource::{CsvDataSource, ParquetDataSource};
use crate::logical_plan::Scan;

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
}
