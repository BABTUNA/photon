//! The DataFrame API (book: "DataFrames").
//!
//! A DataFrame is a handle to an *unexecuted* logical plan plus fluent
//! methods that wrap it in one more node each. Building is free — no data
//! moves until `collect()` arrives (2.9). The SQL frontend (3.11) is just a
//! different syntax producing these same trees.

use std::sync::Arc;

use arrow::datatypes::SchemaRef;

use crate::logical_expr::{AggregateExpr, LogicalExpr};
use crate::logical_plan::{Aggregate, LogicalPlan, Projection, Selection};

pub struct DataFrame {
    plan: Arc<dyn LogicalPlan>,
}

impl DataFrame {
    pub fn new(plan: Arc<dyn LogicalPlan>) -> Self {
        Self { plan }
    }

    /// The SELECT-list: one output column per expression.
    pub fn project(self, exprs: Vec<LogicalExpr>) -> DataFrame {
        DataFrame::new(Arc::new(Projection::new(self.plan, exprs)))
    }

    /// WHERE: keep rows for which `expr` is true.
    pub fn filter(self, expr: LogicalExpr) -> DataFrame {
        DataFrame::new(Arc::new(Selection::new(self.plan, expr)))
    }

    /// GROUP BY `group_exprs`, folding each group through `aggregate_exprs`.
    pub fn aggregate(
        self,
        group_exprs: Vec<LogicalExpr>,
        aggregate_exprs: Vec<AggregateExpr>,
    ) -> DataFrame {
        DataFrame::new(Arc::new(Aggregate::new(
            self.plan,
            group_exprs,
            aggregate_exprs,
        )))
    }

    /// Schema the plan built so far would produce — usable before execution.
    pub fn schema(&self) -> SchemaRef {
        self.plan.schema()
    }

    /// Hand the finished tree to a planner / optimizer / executor.
    pub fn logical_plan(&self) -> Arc<dyn LogicalPlan> {
        Arc::clone(&self.plan)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::ExecutionContext;
    use crate::logical_expr::{col, lit, max};
    use crate::logical_plan::format_plan;
    use arrow::datatypes::DataType;

    #[test]
    fn chain_builds_plan_and_narrows_schema() {
        let ctx = ExecutionContext::new();
        let df = ctx
            .csv("testdata/employee.csv")
            .filter(col("state").eq(lit("CO")))
            .project(vec![col("id"), col("first_name")]);

        let schema = df.schema();
        assert_eq!(schema.fields().len(), 2);
        assert_eq!(schema.field(0).name(), "id");
        assert_eq!(schema.field(1).name(), "first_name");

        // Projection -> Selection -> Scan, each with exactly one child.
        let plan = df.logical_plan();
        let sel = &plan.children()[0];
        let scan = &sel.children()[0];
        assert!(scan.children().is_empty());
        assert!(format_plan(plan.as_ref()).starts_with("Projection: #id, #first_name\n"));
    }

    #[test]
    fn aggregate_frame_derives_group_then_agg_schema() {
        let ctx = ExecutionContext::new();
        let df = ctx
            .csv("testdata/employee.csv")
            .aggregate(vec![col("state")], vec![max(col("salary"))]);

        let schema = df.schema();
        assert_eq!(schema.field(0).name(), "state");
        assert_eq!(schema.field(1).name(), "MAX");
        assert_eq!(schema.field(1).data_type(), &DataType::Int64);
    }

    #[test]
    fn parquet_frames_build_the_same_way() {
        use arrow::array::Int64Array;
        use arrow::datatypes::{Field, Schema};
        use arrow::record_batch::RecordBatch as ArrowRecordBatch;
        use std::fs::File;

        let schema = Arc::new(Schema::new(vec![Field::new("n", DataType::Int64, false)]));
        let batch = ArrowRecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Int64Array::from(vec![1, 2, 3]))],
        )
        .unwrap();
        let path = std::env::temp_dir().join("engine_df_test.parquet");
        let mut writer =
            parquet::arrow::ArrowWriter::try_new(File::create(&path).unwrap(), schema, None)
                .unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();

        let ctx = ExecutionContext::new();
        let df = ctx
            .parquet(path.to_str().unwrap())
            .filter(col("n").gt(lit(1i64)));
        assert_eq!(df.schema().fields().len(), 1);
        assert_eq!(df.schema().field(0).name(), "n");
    }
}
