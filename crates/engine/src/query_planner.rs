//! The query planner (book: "Query Planning").
//!
//! Translates WHAT into HOW: walks the logical tree bottom-up and picks a
//! physical implementation for every node and expression. This is also
//! where column NAMES die — each expression is resolved to an INDEX against
//! its input's schema, exactly once.

use std::sync::Arc;

use crate::logical_expr::LogicalExpr;
use crate::logical_plan::{Aggregate, LogicalPlan, Projection, Scan, Selection};
use crate::physical_expr::{
    AggregateExpression, BinaryExpression, ColumnExpression, Expression, LiteralExpression,
};
use crate::physical_plan::{HashAggregateExec, PhysicalPlan, ProjectionExec, ScanExec, SelectionExec};

/// Translate a logical plan into a runnable physical plan.
pub fn create_physical_plan(plan: &dyn LogicalPlan) -> Arc<dyn PhysicalPlan> {
    let any = plan.as_any();
    if let Some(scan) = any.downcast_ref::<Scan>() {
        Arc::new(ScanExec::new(
            Arc::clone(&scan.data_source),
            scan.projection.clone(),
        ))
    } else if let Some(projection) = any.downcast_ref::<Projection>() {
        let input = create_physical_plan(projection.input.as_ref());
        let exprs = projection
            .exprs
            .iter()
            .map(|e| create_physical_expr(e, projection.input.as_ref()))
            .collect();
        // The physical node takes the LOGICAL node's derived schema —
        // single source of truth for what this operator produces.
        Arc::new(ProjectionExec::new(input, projection.schema(), exprs))
    } else if let Some(selection) = any.downcast_ref::<Selection>() {
        let input = create_physical_plan(selection.input.as_ref());
        let expr = create_physical_expr(&selection.expr, selection.input.as_ref());
        Arc::new(SelectionExec::new(input, expr))
    } else if let Some(aggregate) = any.downcast_ref::<Aggregate>() {
        let input = create_physical_plan(aggregate.input.as_ref());
        let group_exprs = aggregate
            .group_exprs
            .iter()
            .map(|e| create_physical_expr(e, aggregate.input.as_ref()))
            .collect();
        let aggregate_exprs = aggregate
            .aggregate_exprs
            .iter()
            .map(|a| AggregateExpression {
                func: a.func,
                expr: create_physical_expr(&a.expr, aggregate.input.as_ref()),
            })
            .collect();
        Arc::new(HashAggregateExec::new(
            input,
            aggregate.schema(),
            group_exprs,
            aggregate_exprs,
        ))
    } else {
        panic!("query planner: unknown logical plan node: {plan}")
    }
}

/// Translate one logical expression against the schema of `input` (the
/// logical node it will read rows from).
pub fn create_physical_expr(expr: &LogicalExpr, input: &dyn LogicalPlan) -> Arc<dyn Expression> {
    match expr {
        LogicalExpr::Column(name) => {
            let index = input
                .schema()
                .index_of(name)
                .unwrap_or_else(|_| panic!("no column named {name:?} in input schema"));
            Arc::new(ColumnExpression { index })
        }
        LogicalExpr::Literal(value) => Arc::new(LiteralExpression {
            value: value.clone(),
        }),
        LogicalExpr::BinaryExpr { left, op, right } => Arc::new(BinaryExpression {
            left: create_physical_expr(left, input),
            op: *op,
            right: create_physical_expr(right, input),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::ExecutionContext;
    use crate::logical_expr::{col, lit, max};

    #[test]
    fn plans_scan_filter_project() {
        let df = ExecutionContext::new()
            .csv("testdata/employee.csv")
            .filter(col("state").eq(lit("CO")))
            .project(vec![col("id"), col("salary").multiply(lit(2i64))]);

        let physical = create_physical_plan(df.logical_plan().as_ref());

        // Names became indices: state=#3, id=#0, salary=#5.
        assert_eq!(physical.to_string(), "ProjectionExec: #0, #5 * 2");
        let selection = &physical.children()[0];
        assert_eq!(selection.to_string(), "SelectionExec: #3 = 'CO'");
        let scan = &selection.children()[0];
        assert_eq!(scan.to_string(), "ScanExec: projection=[]");
        assert!(scan.children().is_empty());

        // Physical schema comes from the logical plan.
        assert_eq!(physical.schema().field(1).name(), "mult");
    }

    #[test]
    fn plans_grouped_aggregate() {
        let df = ExecutionContext::new()
            .csv("testdata/employee.csv")
            .aggregate(vec![col("state")], vec![max(col("salary"))]);

        let physical = create_physical_plan(df.logical_plan().as_ref());
        assert_eq!(
            physical.to_string(),
            "HashAggregateExec: groupExpr=[#3], aggregateExpr=[MAX(#5)]"
        );
        assert_eq!(physical.schema().field(0).name(), "state");
        assert_eq!(physical.schema().field(1).name(), "MAX");
    }

    #[test]
    #[should_panic(expected = "no column named")]
    fn planning_rejects_unknown_columns() {
        let df = ExecutionContext::new()
            .csv("testdata/employee.csv")
            .filter(col("bogus").eq(lit(1i64)));
        create_physical_plan(df.logical_plan().as_ref());
    }
}
