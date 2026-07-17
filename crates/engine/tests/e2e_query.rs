//! End-to-end queries: DataFrame → logical plan → planner → physical plan →
//! results. The first tests where every layer of the engine runs at once.
//!
//! employee.csv: (1 Bill CA 12000) (2 Gregg CO 10000) (3 John CO 11500)
//! (4 Von NULL 11800).

use engine::dataframe::DataFrame;
use engine::datatypes::{RecordBatch, ScalarValue};
use engine::execution::ExecutionContext;
use engine::logical_expr::{avg, col, count, lit, max, sum};

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

/// Group output order is nondeterministic (hash map); find rows by key.
fn row_by_key(batch: &RecordBatch, key: Option<ScalarValue>) -> Vec<Option<ScalarValue>> {
    let row = (0..batch.row_count())
        .find(|&i| batch.column(0).value(i) == key)
        .unwrap_or_else(|| panic!("no group with key {key:?}"));
    (0..batch.column_count())
        .map(|c| batch.column(c).value(row))
        .collect()
}

#[test]
fn grouped_aggregate_end_to_end() {
    let df = ExecutionContext::new()
        .csv("testdata/employee.csv")
        .aggregate(
            vec![col("state")],
            vec![max(col("salary")), count(col("id"))],
        );

    let batches = run(&df);
    assert_eq!(batches.len(), 1);
    let batch = &batches[0];
    assert_eq!(batch.row_count(), 3); // CA, CO, and the NULL state

    assert_eq!(
        row_by_key(batch, text("CA")),
        vec![text("CA"), int(12000), int(1)]
    );
    assert_eq!(
        row_by_key(batch, text("CO")),
        vec![text("CO"), int(11500), int(2)]
    );
    // SQL GROUP BY puts all NULL keys in ONE group.
    assert_eq!(row_by_key(batch, None), vec![None, int(11800), int(1)]);
}

#[test]
fn global_aggregate_without_group_by() {
    let df = ExecutionContext::new().csv("testdata/employee.csv").aggregate(
        vec![],
        vec![sum(col("salary")), avg(col("salary")), count(col("state"))],
    );

    let batches = run(&df);
    let batch = &batches[0];
    assert_eq!(batch.row_count(), 1);
    assert_eq!(batch.column(0).value(0), int(45300));
    assert_eq!(batch.column(1).value(0), Some(ScalarValue::Float64(11325.0)));
    // COUNT(state) skips the NULL: 3, not 4.
    assert_eq!(batch.column(2).value(0), int(3));
}

#[test]
fn filter_feeds_aggregate() {
    // Only salaries >= 11000 reach the aggregate.
    let df = ExecutionContext::new()
        .csv("testdata/employee.csv")
        .filter(col("salary").gt_eq(lit(11000i64)))
        .aggregate(vec![], vec![count(col("id")), max(col("salary"))]);

    let batch = &run(&df)[0];
    assert_eq!(batch.column(0).value(0), int(3));
    assert_eq!(batch.column(1).value(0), int(12000));
}
