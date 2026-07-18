//! Two-table join, end to end, on a TPC-H sample (nation ⋈ region).
//!
//! region: AFRICA(0), AMERICA(1), ASIA(2). nation: three American nations,
//! two Asian, none African — AFRICA only survives outer joins.

use engine::datatypes::{RecordBatch, ScalarValue};
use engine::execution::ExecutionContext;
use engine::logical_expr::{col, count};
use engine::logical_plan::JoinType;

fn text(v: &str) -> Option<ScalarValue> {
    Some(ScalarValue::Utf8(v.to_string()))
}

/// All rows of a batch, sorted for order-insensitive comparison.
fn sorted_rows(batches: &[RecordBatch]) -> Vec<Vec<Option<ScalarValue>>> {
    let mut rows: Vec<Vec<_>> = batches
        .iter()
        .flat_map(|b| {
            (0..b.row_count())
                .map(move |i| (0..b.column_count()).map(|c| b.column(c).value(i)).collect())
        })
        .collect();
    rows.sort_by_key(|r| format!("{r:?}"));
    rows
}

#[test]
fn inner_join_nation_to_region() {
    let ctx = ExecutionContext::new();
    let nation = ctx.csv("testdata/nation.csv");
    let region = ctx.csv("testdata/region.csv");

    let df = nation
        .join(&region, JoinType::Inner, vec![("n_regionkey", "r_regionkey")])
        .project(vec![col("n_name"), col("r_name")]);

    // Joined schema feeds the projection; 5 nations all match a region.
    assert_eq!(df.schema().fields().len(), 2);
    assert_eq!(
        sorted_rows(&ctx.collect(&df)),
        vec![
            vec![text("ARGENTINA"), text("AMERICA")],
            vec![text("BRAZIL"), text("AMERICA")],
            vec![text("CANADA"), text("AMERICA")],
            vec![text("INDIA"), text("ASIA")],
            vec![text("JAPAN"), text("ASIA")],
        ]
    );
}

#[test]
fn left_join_keeps_nationless_regions() {
    let ctx = ExecutionContext::new();
    let region = ctx.csv("testdata/region.csv");
    let nation = ctx.csv("testdata/nation.csv");

    let df = region
        .join(&nation, JoinType::Left, vec![("r_regionkey", "n_regionkey")])
        .project(vec![col("r_name"), col("n_name")]);

    // 3 + 2 matches plus AFRICA padded with NULL.
    assert_eq!(
        sorted_rows(&ctx.collect(&df)),
        vec![
            vec![text("AFRICA"), None],
            vec![text("AMERICA"), text("ARGENTINA")],
            vec![text("AMERICA"), text("BRAZIL")],
            vec![text("AMERICA"), text("CANADA")],
            vec![text("ASIA"), text("INDIA")],
            vec![text("ASIA"), text("JAPAN")],
        ]
    );
}

#[test]
fn join_feeds_aggregate_nations_per_region() {
    let ctx = ExecutionContext::new();
    let nation = ctx.csv("testdata/nation.csv");
    let region = ctx.csv("testdata/region.csv");

    let df = nation
        .join(&region, JoinType::Inner, vec![("n_regionkey", "r_regionkey")])
        .aggregate(vec![col("r_name")], vec![count(col("n_nationkey"))]);

    assert_eq!(
        sorted_rows(&ctx.collect(&df)),
        vec![
            vec![text("AMERICA"), Some(ScalarValue::Int64(3))],
            vec![text("ASIA"), Some(ScalarValue::Int64(2))],
        ]
    );
}
