//! Golden tests: DataFrame chains must produce exactly the expected logical
//! plan tree. If a Display impl, builder, or node ordering changes shape,
//! the diff shows up here as a failing string comparison.

use engine::dataframe::DataFrame;
use engine::execution::ExecutionContext;
use engine::logical_expr::{avg, col, count, lit, max};
use engine::logical_plan::format_plan;

fn plan_of(df: DataFrame) -> String {
    format_plan(df.logical_plan().as_ref())
}

#[test]
fn filter_then_project_matches_the_book() {
    let df = ExecutionContext::new()
        .csv("testdata/employee.csv")
        .filter(col("state").eq(lit("CO")))
        .project(vec![col("id"), col("first_name"), col("last_name")]);

    assert_eq!(
        plan_of(df),
        "Projection: #id, #first_name, #last_name\n\
         \tSelection: #state = 'CO'\n\
         \t\tScan: testdata/employee.csv; projection=None\n"
    );
}

#[test]
fn grouped_aggregate_plan() {
    let df = ExecutionContext::new()
        .csv("testdata/employee.csv")
        .aggregate(
            vec![col("state")],
            vec![max(col("salary")), count(col("id"))],
        );

    assert_eq!(
        plan_of(df),
        "Aggregate: groupExpr=[#state], aggregateExpr=[MAX(#salary), COUNT(#id)]\n\
         \tScan: testdata/employee.csv; projection=None\n"
    );
}

#[test]
fn filter_aggregate_project_pipeline() {
    let df = ExecutionContext::new()
        .csv("testdata/employee.csv")
        .filter(col("salary").gt_eq(lit(10000i64)))
        .aggregate(vec![col("state")], vec![avg(col("salary").multiply(lit(2i64)))])
        .project(vec![col("state")]);

    assert_eq!(
        plan_of(df),
        "Projection: #state\n\
         \tAggregate: groupExpr=[#state], aggregateExpr=[AVG(#salary * 2)]\n\
         \t\tSelection: #salary >= 10000\n\
         \t\t\tScan: testdata/employee.csv; projection=None\n"
    );
}
