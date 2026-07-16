//! End-to-end queries: DataFrame → logical plan → planner → physical plan →
//! results. The first tests where every layer of the engine runs at once.
//!
//! employee.csv: (1 Bill CA 12000) (2 Gregg CO 10000) (3 John CO 11500)
//! (4 Von NULL 11800).

use engine::dataframe::DataFrame;
use engine::datatypes::{RecordBatch, ScalarValue};
use engine::execution::ExecutionContext;
use engine::logical_expr::{col, lit};

fn int(v: i64) -> Option<ScalarValue> {
    Some(ScalarValue::Int64(v))
}

fn text(v: &str) -> Option<ScalarValue> {
    Some(ScalarValue::Utf8(v.to_string()))
}

/// Flatten one column of every batch, in order.
fn column_values(batches: &[RecordBatch], col: usize) -> Vec<Option<ScalarValue>> {
    batches
        .iter()
        .flat_map(|b| (0..b.row_count()).map(move |i| b.column(col).value(i)))
        .collect()
}

fn run(df: &DataFrame) -> Vec<RecordBatch> {
    ExecutionContext::new().collect(df)
}

#[test]
fn scan_filter_project_the_books_first_query() {
    let df = ExecutionContext::new()
        .csv("testdata/employee.csv")
        .filter(col("state").eq(lit("CO")))
        .project(vec![col("id"), col("first_name")]);

    let batches = run(&df);
    let schema = df.schema();
    assert_eq!(schema.field(0).name(), "id");
    assert_eq!(schema.field(1).name(), "first_name");

    assert_eq!(column_values(&batches, 0), vec![int(2), int(3)]);
    assert_eq!(column_values(&batches, 1), vec![text("Gregg"), text("John")]);
}

#[test]
fn projection_computes_math_over_every_row() {
    let df = ExecutionContext::new()
        .csv("testdata/employee.csv")
        .project(vec![col("id"), col("salary").multiply(lit(2i64))]);

    let batches = run(&df);
    assert_eq!(
        column_values(&batches, 1),
        vec![int(24000), int(20000), int(23000), int(23600)]
    );
}

#[test]
fn filter_on_a_computed_predicate() {
    // salary > 11000 keeps rows 1, 3, 4 — including the NULL-state row.
    let df = ExecutionContext::new()
        .csv("testdata/employee.csv")
        .filter(col("salary").gt(lit(11000i64)))
        .project(vec![col("first_name"), col("state")]);

    let batches = run(&df);
    assert_eq!(
        column_values(&batches, 0),
        vec![text("Bill"), text("John"), text("Von")]
    );
    // NULL passes through projection untouched.
    assert_eq!(column_values(&batches, 1), vec![text("CA"), text("CO"), None]);
}
