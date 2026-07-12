//! The data source contract (book: "Data Sources").
//!
//! Everything the engine reads — in-memory tables, CSV, parquet, and later
//! Part 2's multimodal clip table — implements one small trait: describe
//! your schema, and scan yourself into a stream of record batches.

use std::fs::File;
use std::sync::Arc;

use arrow::csv::ReaderBuilder;
use arrow::csv::reader::Format;
use arrow::datatypes::{Schema, SchemaRef};
use parquet::arrow::ProjectionMask;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

use crate::datatypes::{RecordBatch, project_by_name};

/// Map projection names to column indices in `schema`. Panics on unknown names.
fn resolve_indices(schema: &Schema, projection: &[String]) -> Vec<usize> {
    projection
        .iter()
        .map(|name| {
            schema
                .index_of(name)
                .unwrap_or_else(|_| panic!("unknown column {name:?} in projection"))
        })
        .collect()
}

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
        let indices = resolve_indices(&self.schema, projection);
        let schema = Arc::new(self.schema.project(&indices).unwrap());
        Box::new(self.data.clone().into_iter().map(move |batch| {
            let columns = indices.iter().map(|&i| Arc::clone(batch.column(i))).collect();
            RecordBatch::new(Arc::clone(&schema), columns)
        }))
    }
}

/// CSV file source. If no schema is supplied, one is inferred at construction
/// time by sampling the file — CSV carries no type information of its own.
pub struct CsvDataSource {
    path: String,
    schema: SchemaRef,
    batch_size: usize,
}

/// How many records the schema inference pass samples.
const INFER_RECORDS: usize = 1000;

impl CsvDataSource {
    pub fn new(path: impl Into<String>, schema: Option<SchemaRef>, batch_size: usize) -> Self {
        let path = path.into();
        let schema = schema.unwrap_or_else(|| Arc::new(Self::infer_schema(&path)));
        Self {
            path,
            schema,
            batch_size,
        }
    }

    /// Read up to `INFER_RECORDS` rows and let arrow vote on column types
    /// (Int64 → Float64 → Utf8, weakest type that fits every sampled value).
    fn infer_schema(path: &str) -> Schema {
        let mut file = File::open(path).unwrap_or_else(|e| panic!("open {path}: {e}"));
        let format = Format::default().with_header(true);
        let (schema, _rows_read) = format
            .infer_schema(&mut file, Some(INFER_RECORDS))
            .unwrap_or_else(|e| panic!("schema inference for {path}: {e}"));
        schema
    }
}

impl DataSource for CsvDataSource {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn scan(&self, projection: &[String]) -> Box<dyn Iterator<Item = RecordBatch>> {
        let file = File::open(&self.path).unwrap_or_else(|e| panic!("open {}: {e}", self.path));
        let mut builder = ReaderBuilder::new(Arc::clone(&self.schema))
            .with_header(true)
            .with_batch_size(self.batch_size);
        if !projection.is_empty() {
            builder = builder.with_projection(resolve_indices(&self.schema, projection));
        }
        let reader = builder
            .build(file)
            .unwrap_or_else(|e| panic!("csv reader for {}: {e}", self.path));
        Box::new(reader.map(|result| result.expect("csv decode error").into()))
    }
}

/// Parquet file source. Unlike CSV, parquet is self-describing: the file
/// footer holds the schema and per-row-group metadata, so `new` reads only
/// the footer — no data pages.
pub struct ParquetDataSource {
    path: String,
    schema: SchemaRef,
    num_row_groups: usize,
}

impl ParquetDataSource {
    pub fn new(path: impl Into<String>) -> Self {
        let path = path.into();
        let file = File::open(&path).unwrap_or_else(|e| panic!("open {path}: {e}"));
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)
            .unwrap_or_else(|e| panic!("parquet footer of {path}: {e}"));
        let schema = Arc::clone(builder.schema());
        let num_row_groups = builder.metadata().num_row_groups();
        Self {
            path,
            schema,
            num_row_groups,
        }
    }

    /// Row groups are parquet's internal horizontal partitions — and the
    /// engine's unit of parallel (4.1) and distributed (5.2) work.
    pub fn num_row_groups(&self) -> usize {
        self.num_row_groups
    }
}

impl DataSource for ParquetDataSource {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn scan(&self, projection: &[String]) -> Box<dyn Iterator<Item = RecordBatch>> {
        let file = File::open(&self.path).unwrap_or_else(|e| panic!("open {}: {e}", self.path));
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)
            .unwrap_or_else(|e| panic!("parquet footer of {}: {e}", self.path));
        if projection.is_empty() {
            let reader = builder.build().expect("parquet reader");
            return Box::new(reader.map(|result| result.expect("parquet decode error").into()));
        }
        // ProjectionMask is a set: the reader skips unwanted column chunks on
        // disk but returns columns in FILE order, not requested order.
        let indices = resolve_indices(&self.schema, projection);
        let mask = ProjectionMask::roots(builder.parquet_schema(), indices.iter().copied());
        let reader = builder
            .with_projection(mask)
            .build()
            .expect("parquet reader");
        // Restore the requested order per batch (cheap: Arc shuffles only).
        let names = projection.to_vec();
        Box::new(reader.map(move |result| {
            let batch: RecordBatch = result.expect("parquet decode error").into();
            reorder_columns(&batch, &names)
        }))
    }
}

/// Rebuild `batch` with columns in exactly `names` order.
fn reorder_columns(batch: &RecordBatch, names: &[String]) -> RecordBatch {
    let schema = batch.schema();
    let name_strs: Vec<&str> = names.iter().map(String::as_str).collect();
    let projected = Arc::new(project_by_name(&schema, &name_strs).unwrap());
    let columns = names
        .iter()
        .map(|n| Arc::clone(batch.column(schema.index_of(n).unwrap())))
        .collect();
    RecordBatch::new(projected, columns)
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

    #[test]
    fn csv_infers_schema_and_scans() {
        let source = CsvDataSource::new("testdata/employee.csv", None, 1024);

        let schema = source.schema();
        assert_eq!(schema.field(0).name(), "id");
        assert_eq!(schema.field(0).data_type(), &DataType::Int64);
        assert_eq!(schema.field(1).data_type(), &DataType::Utf8);
        assert_eq!(schema.field(5).name(), "salary");
        assert_eq!(schema.field(5).data_type(), &DataType::Int64);

        let batches: Vec<_> = source.scan(&[]).collect();
        assert_eq!(batches.len(), 1);
        let batch = &batches[0];
        assert_eq!(batch.row_count(), 4);
        // Quoted comma survives parsing; empty field becomes NULL.
        assert_eq!(
            batch.column(4).value(2),
            Some(ScalarValue::Utf8("Manager, Software".to_string()))
        );
        assert_eq!(batch.column(3).value(3), None);
    }

    /// Write a 5-row parquet file capped at 2 rows per row group → 3 groups.
    fn write_test_parquet(name: &str) -> String {
        use parquet::arrow::ArrowWriter;
        use parquet::file::properties::WriterProperties;

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("city", DataType::Utf8, false),
            Field::new("temp_c", DataType::Float64, false),
        ]));
        let batch = ArrowRecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int64Array::from(vec![1, 2, 3, 4, 5])),
                Arc::new(StringArray::from(vec!["a", "b", "c", "d", "e"])),
                Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0, 5.0])),
            ],
        )
        .unwrap();

        let path = std::env::temp_dir().join(name);
        let file = File::create(&path).unwrap();
        let props = WriterProperties::builder()
            .set_max_row_group_row_count(Some(2))
            .build();
        let mut writer = ArrowWriter::try_new(file, schema, Some(props)).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
        path.to_str().unwrap().to_string()
    }

    #[test]
    fn parquet_reads_schema_and_row_groups_from_footer() {
        let path = write_test_parquet("engine_test_footer.parquet");
        let source = ParquetDataSource::new(&path);

        assert_eq!(source.schema().field(0).name(), "id");
        assert_eq!(source.schema().field(2).data_type(), &DataType::Float64);
        assert_eq!(source.num_row_groups(), 3); // 5 rows / max 2 per group
    }

    #[test]
    fn parquet_full_scan_yields_all_rows_across_row_groups() {
        let path = write_test_parquet("engine_test_scan.parquet");
        let source = ParquetDataSource::new(&path);

        let batches: Vec<_> = source.scan(&[]).collect();
        let total: usize = batches.iter().map(|b| b.row_count()).sum();
        assert_eq!(total, 5);
        assert_eq!(batches[0].column_count(), 3);
    }

    #[test]
    fn parquet_projection_skips_and_reorders_columns() {
        let path = write_test_parquet("engine_test_proj.parquet");
        let source = ParquetDataSource::new(&path);

        let batches: Vec<_> = source
            .scan(&["temp_c".to_string(), "id".to_string()])
            .collect();

        let batch = &batches[0];
        assert_eq!(batch.column_count(), 2);
        assert_eq!(batch.schema().field(0).name(), "temp_c");
        assert_eq!(batch.schema().field(1).name(), "id");
        assert_eq!(batch.column(0).value(0), Some(ScalarValue::Float64(1.0)));
        assert_eq!(batch.column(1).value(0), Some(ScalarValue::Int64(1)));
    }

    #[test]
    fn csv_scan_with_projection() {
        let source = CsvDataSource::new("testdata/employee.csv", None, 1024);
        let batches: Vec<_> = source
            .scan(&["salary".to_string(), "id".to_string()])
            .collect();

        let batch = &batches[0];
        assert_eq!(batch.column_count(), 2);
        assert_eq!(batch.schema().field(0).name(), "salary");
        assert_eq!(batch.schema().field(1).name(), "id");
        assert_eq!(batch.column(0).value(0), Some(ScalarValue::Int64(12000)));
        assert_eq!(batch.column(1).value(0), Some(ScalarValue::Int64(1)));
    }
}
