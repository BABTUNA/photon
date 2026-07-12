//! Integration tests: the data source layer exercised from outside the
//! crate, through the public API only — CSV and parquet must be
//! interchangeable behind `dyn DataSource`.

use std::fs::File;
use std::sync::Arc;

use engine::datasource::{CsvDataSource, DataSource, ParquetDataSource};
use engine::datatypes::ScalarValue;

/// Convert the employee.csv fixture into a parquet file so both sources
/// hold identical data.
fn employee_parquet() -> String {
    let mut csv = File::open("testdata/employee.csv").unwrap();
    let format = arrow::csv::reader::Format::default().with_header(true);
    let (schema, _) = format.infer_schema(&mut csv, None).unwrap();
    let csv = File::open("testdata/employee.csv").unwrap();
    let reader = arrow::csv::ReaderBuilder::new(Arc::new(schema))
        .with_header(true)
        .build(csv)
        .unwrap();

    let path = std::env::temp_dir().join("engine_it_employee.parquet");
    let out = File::create(&path).unwrap();
    let mut writer = None;
    for batch in reader {
        let batch = batch.unwrap();
        let w = writer.get_or_insert_with(|| {
            parquet::arrow::ArrowWriter::try_new(out.try_clone().unwrap(), batch.schema(), None)
                .unwrap()
        });
        w.write(&batch).unwrap();
    }
    writer.unwrap().close().unwrap();
    path.to_str().unwrap().to_string()
}

fn sources() -> Vec<(&'static str, Box<dyn DataSource>)> {
    vec![
        (
            "csv",
            Box::new(CsvDataSource::new("testdata/employee.csv", None, 1024)) as Box<dyn DataSource>,
        ),
        (
            "parquet",
            Box::new(ParquetDataSource::new(employee_parquet())) as Box<dyn DataSource>,
        ),
    ]
}

#[test]
fn both_sources_expose_the_same_schema() {
    for (kind, source) in sources() {
        let schema = source.schema();
        let names: Vec<_> = schema.fields().iter().map(|f| f.name().clone()).collect();
        assert_eq!(
            names,
            ["id", "first_name", "last_name", "state", "job_title", "salary"],
            "schema mismatch for {kind}"
        );
    }
}

#[test]
fn full_scan_returns_every_row_and_column() {
    for (kind, source) in sources() {
        let batches: Vec<_> = source.scan(&[]).collect();
        let rows: usize = batches.iter().map(|b| b.row_count()).sum();
        assert_eq!(rows, 4, "row count mismatch for {kind}");
        assert_eq!(batches[0].column_count(), 6, "column count mismatch for {kind}");
    }
}

#[test]
fn projected_scan_agrees_across_formats() {
    let projection = ["salary".to_string(), "first_name".to_string()];
    let mut results: Vec<Vec<Option<ScalarValue>>> = vec![];

    for (kind, source) in sources() {
        let batches: Vec<_> = source.scan(&projection).collect();

        // Projection is honored: only the two columns, in requested order.
        assert_eq!(batches[0].column_count(), 2, "for {kind}");
        assert_eq!(batches[0].schema().field(0).name(), "salary", "for {kind}");
        assert_eq!(batches[0].schema().field(1).name(), "first_name", "for {kind}");

        // Flatten all values row-major for cross-format comparison.
        let mut values = vec![];
        for batch in &batches {
            for row in 0..batch.row_count() {
                values.push(batch.column(0).value(row));
                values.push(batch.column(1).value(row));
            }
        }
        results.push(values);
    }

    assert_eq!(results[0], results[1], "csv and parquet disagree");
    assert_eq!(results[0][0], Some(ScalarValue::Int64(12000)));
    assert_eq!(results[0][1], Some(ScalarValue::Utf8("Bill".to_string())));
}
