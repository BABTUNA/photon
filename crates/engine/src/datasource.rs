//! The data source contract (book: "Data Sources").
//!
//! Everything the engine reads — in-memory tables, CSV, parquet, and later
//! Part 2's multimodal clip table — implements one small trait: describe
//! your schema, and scan yourself into a stream of record batches.

use std::sync::Arc;

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

/// The simplest source: batches already sitting in memory. The test double
/// for everything above this layer, and the shape shuffled partitions arrive
/// in on the distributed path (5.12).
pub struct InMemoryDataSource {
    schema: SchemaRef,
    data: Vec<RecordBatch>,
}

impl InMemoryDataSource {
    pub fn new(schema: SchemaRef, data: Vec<RecordBatch>) -> Self {
        Self { schema, data }
    }
}

impl DataSource for InMemoryDataSource {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn scan(&self, projection: &[String]) -> Box<dyn Iterator<Item = RecordBatch>> {
        if projection.is_empty() {
            return Box::new(self.data.clone().into_iter());
        }
        // Resolve names once, not per batch.
        let indices: Vec<usize> = projection
            .iter()
            .map(|name| {
                self.schema
                    .index_of(name)
                    .unwrap_or_else(|_| panic!("unknown column {name:?} in projection"))
            })
            .collect();
        let schema = Arc::new(self.schema.project(&indices).unwrap());
        Box::new(self.data.clone().into_iter().map(move |batch| {
            let columns = indices.iter().map(|&i| Arc::clone(batch.column(i))).collect();
            RecordBatch::new(Arc::clone(&schema), columns)
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datatypes::ScalarValue;
    use arrow::array::{Float64Array, Int64Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch as ArrowRecordBatch;

    fn test_source() -> InMemoryDataSource {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int64, false),
            Field::new("b", DataType::Utf8, false),
            Field::new("c", DataType::Float64, false),
        ]));
        let batch = |ids: Vec<i64>, names: Vec<&str>, vals: Vec<f64>| {
            ArrowRecordBatch::try_new(
                Arc::clone(&schema),
                vec![
                    Arc::new(Int64Array::from(ids)),
                    Arc::new(StringArray::from(names)),
                    Arc::new(Float64Array::from(vals)),
                ],
            )
            .unwrap()
            .into()
        };
        let data = vec![
            batch(vec![1, 2], vec!["x", "y"], vec![1.0, 2.0]),
            batch(vec![3], vec!["z"], vec![3.0]),
        ];
        InMemoryDataSource::new(schema, data)
    }

    #[test]
    fn empty_projection_scans_everything() {
        let source = test_source();
        let batches: Vec<_> = source.scan(&[]).collect();

        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].column_count(), 3);
        assert_eq!(batches[0].row_count(), 2);
        assert_eq!(batches[1].column(0).value(0), Some(ScalarValue::Int64(3)));
    }

    #[test]
    fn projection_subsets_and_reorders_columns() {
        let source = test_source();
        let batches: Vec<_> = source.scan(&["c".to_string(), "a".to_string()]).collect();

        let first = &batches[0];
        assert_eq!(first.column_count(), 2);
        assert_eq!(first.schema().field(0).name(), "c");
        assert_eq!(first.schema().field(1).name(), "a");
        assert_eq!(first.column(0).value(1), Some(ScalarValue::Float64(2.0)));
        assert_eq!(first.column(1).value(1), Some(ScalarValue::Int64(2)));
    }

    #[test]
    #[should_panic(expected = "unknown column")]
    fn unknown_projection_column_panics() {
        let source = test_source();
        // Names resolve eagerly, so this panics before the iterator is touched.
        let _ = source.scan(&["nope".to_string()]);
    }
}
