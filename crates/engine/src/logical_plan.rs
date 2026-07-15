//! Logical plans (book: "Logical Plans").
//!
//! A logical plan is a tree describing WHAT a query computes — which data,
//! which columns, which filters — with no opinion on HOW. The how (hash vs
//! sort, serial vs parallel) is the physical plan's job (commit 2.3). Every
//! frontend (DataFrame in 2.1, SQL in 3.10) produces this same tree.

use std::fmt::Display;
use std::sync::Arc;

use arrow::datatypes::SchemaRef;

use arrow::datatypes::Schema;

use crate::datasource::DataSource;
use crate::datatypes::project_by_name;
use crate::logical_expr::{AggregateExpr, LogicalExpr};

/// One node in the tree. `Display` is a supertrait so every node can print
/// itself on one line — the pretty-printer (1.15) walks the tree with it.
pub trait LogicalPlan: Display + Send + Sync {
    /// Schema of the rows this node produces.
    fn schema(&self) -> SchemaRef;

    /// The nodes this one consumes. Empty for leaves.
    fn children(&self) -> Vec<Arc<dyn LogicalPlan>>;
}

/// Leaf node: read `projection` columns from a data source. The only node
/// that has no children — every plan tree bottoms out in Scans.
pub struct Scan {
    pub path: String,
    pub data_source: Arc<dyn DataSource>,
    pub projection: Vec<String>,
    schema: SchemaRef,
}

impl Scan {
    pub fn new(
        path: impl Into<String>,
        data_source: Arc<dyn DataSource>,
        projection: Vec<String>,
    ) -> Self {
        // Derived once here: the schema this Scan PRODUCES (post-projection),
        // not the full schema the source HAS.
        let schema = if projection.is_empty() {
            data_source.schema()
        } else {
            let names: Vec<&str> = projection.iter().map(String::as_str).collect();
            Arc::new(project_by_name(&data_source.schema(), &names).unwrap())
        };
        Self {
            path: path.into(),
            data_source,
            projection,
            schema,
        }
    }
}

impl Display for Scan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.projection.is_empty() {
            write!(f, "Scan: {}; projection=None", self.path)
        } else {
            write!(f, "Scan: {}; projection={:?}", self.path, self.projection)
        }
    }
}

impl LogicalPlan for Scan {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn children(&self) -> Vec<Arc<dyn LogicalPlan>> {
        vec![]
    }
}

/// Compute new columns from the input: `SELECT a, b * 2`.
/// Output schema = one field per expression.
pub struct Projection {
    pub input: Arc<dyn LogicalPlan>,
    pub exprs: Vec<LogicalExpr>,
    schema: SchemaRef,
}

impl Projection {
    pub fn new(input: Arc<dyn LogicalPlan>, exprs: Vec<LogicalExpr>) -> Self {
        let fields: Vec<_> = exprs.iter().map(|e| e.to_field(input.as_ref())).collect();
        let schema = Arc::new(Schema::new(fields));
        Self {
            input,
            exprs,
            schema,
        }
    }
}

impl Display for Projection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Projection: {}", join(&self.exprs))
    }
}

impl LogicalPlan for Projection {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn children(&self) -> Vec<Arc<dyn LogicalPlan>> {
        vec![Arc::clone(&self.input)]
    }
}

/// Keep only rows where `expr` is true: `WHERE state = 'CO'`.
/// Filters change the row count, never the shape — schema passes through.
pub struct Selection {
    pub input: Arc<dyn LogicalPlan>,
    pub expr: LogicalExpr,
}

impl Selection {
    pub fn new(input: Arc<dyn LogicalPlan>, expr: LogicalExpr) -> Self {
        Self { input, expr }
    }
}

impl Display for Selection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Selection: {}", self.expr)
    }
}

impl LogicalPlan for Selection {
    fn schema(&self) -> SchemaRef {
        self.input.schema()
    }

    fn children(&self) -> Vec<Arc<dyn LogicalPlan>> {
        vec![Arc::clone(&self.input)]
    }
}

/// Group rows by `group_exprs` and fold each group through the aggregates:
/// `SELECT state, MAX(salary) ... GROUP BY state`.
/// Output schema = group fields, then aggregate fields.
pub struct Aggregate {
    pub input: Arc<dyn LogicalPlan>,
    pub group_exprs: Vec<LogicalExpr>,
    pub aggregate_exprs: Vec<AggregateExpr>,
    schema: SchemaRef,
}

impl Aggregate {
    pub fn new(
        input: Arc<dyn LogicalPlan>,
        group_exprs: Vec<LogicalExpr>,
        aggregate_exprs: Vec<AggregateExpr>,
    ) -> Self {
        let fields: Vec<_> = group_exprs
            .iter()
            .map(|e| e.to_field(input.as_ref()))
            .chain(aggregate_exprs.iter().map(|a| a.to_field(input.as_ref())))
            .collect();
        let schema = Arc::new(Schema::new(fields));
        Self {
            input,
            group_exprs,
            aggregate_exprs,
            schema,
        }
    }
}

impl Display for Aggregate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Aggregate: groupExpr=[{}], aggregateExpr=[{}]",
            join(&self.group_exprs),
            join(&self.aggregate_exprs)
        )
    }
}

impl LogicalPlan for Aggregate {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn children(&self) -> Vec<Arc<dyn LogicalPlan>> {
        vec![Arc::clone(&self.input)]
    }
}

fn join(items: &[impl Display]) -> String {
    items
        .iter()
        .map(|i| i.to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

/// Render a plan as the book's indented tree, children one tab deeper:
///
/// ```text
/// Projection: #id, #first_name
///     Selection: #state = 'CO'
///         Scan: employee; projection=None
/// ```
pub fn format_plan(plan: &dyn LogicalPlan) -> String {
    fn walk(plan: &dyn LogicalPlan, indent: usize, out: &mut String) {
        out.push_str(&"\t".repeat(indent));
        out.push_str(&plan.to_string());
        out.push('\n');
        for child in plan.children() {
            walk(child.as_ref(), indent + 1, out);
        }
    }
    let mut out = String::new();
    walk(plan, 0, &mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datasource::CsvDataSource;
    use arrow::datatypes::DataType;

    fn employee_source() -> Arc<dyn DataSource> {
        Arc::new(CsvDataSource::new("testdata/employee.csv", None, 1024))
    }

    #[test]
    fn scan_without_projection_exposes_source_schema() {
        let scan = Scan::new("employee", employee_source(), vec![]);

        assert_eq!(scan.schema().fields().len(), 6);
        assert!(scan.children().is_empty());
        assert_eq!(scan.to_string(), "Scan: employee; projection=None");
    }

    #[test]
    fn scan_with_projection_narrows_schema() {
        let scan = Scan::new(
            "employee",
            employee_source(),
            vec!["salary".to_string(), "state".to_string()],
        );

        let schema = scan.schema();
        assert_eq!(schema.fields().len(), 2);
        assert_eq!(schema.field(0).name(), "salary");
        assert_eq!(schema.field(0).data_type(), &DataType::Int64);
        assert_eq!(
            scan.to_string(),
            r#"Scan: employee; projection=["salary", "state"]"#
        );
    }

    #[test]
    fn scan_works_as_a_trait_object() {
        let plan: Arc<dyn LogicalPlan> =
            Arc::new(Scan::new("employee", employee_source(), vec![]));
        assert_eq!(plan.schema().field(0).name(), "id");
    }

    #[test]
    fn projection_derives_schema_from_exprs() {
        use crate::logical_expr::{col, lit};

        let scan: Arc<dyn LogicalPlan> = Arc::new(Scan::new("employee", employee_source(), vec![]));
        let proj = Projection::new(
            scan,
            vec![col("id"), col("salary").multiply(lit(2i64))],
        );

        let schema = proj.schema();
        assert_eq!(schema.field(0).name(), "id");
        assert_eq!(schema.field(1).name(), "mult");
        assert_eq!(schema.field(1).data_type(), &DataType::Int64);
        assert_eq!(proj.to_string(), "Projection: #id, #salary * 2");
    }

    #[test]
    fn selection_passes_schema_through_unchanged() {
        use crate::logical_expr::{col, lit};

        let scan: Arc<dyn LogicalPlan> = Arc::new(Scan::new("employee", employee_source(), vec![]));
        let sel = Selection::new(Arc::clone(&scan), col("state").eq(lit("CO")));

        assert_eq!(sel.schema().fields().len(), 6);
        assert_eq!(sel.to_string(), "Selection: #state = 'CO'");
    }

    #[test]
    fn aggregate_schema_is_groups_then_aggregates() {
        use crate::logical_expr::{col, max};

        let scan: Arc<dyn LogicalPlan> = Arc::new(Scan::new("employee", employee_source(), vec![]));
        let agg = Aggregate::new(scan, vec![col("state")], vec![max(col("salary"))]);

        let schema = agg.schema();
        assert_eq!(schema.field(0).name(), "state");
        assert_eq!(schema.field(1).name(), "MAX");
        assert_eq!(schema.field(1).data_type(), &DataType::Int64);
        assert_eq!(
            agg.to_string(),
            "Aggregate: groupExpr=[#state], aggregateExpr=[MAX(#salary)]"
        );
    }

    #[test]
    fn format_plan_prints_the_book_tree() {
        use crate::logical_expr::{col, lit};

        let scan: Arc<dyn LogicalPlan> = Arc::new(Scan::new("employee", employee_source(), vec![]));
        let filtered: Arc<dyn LogicalPlan> =
            Arc::new(Selection::new(scan, col("state").eq(lit("CO"))));
        let plan = Projection::new(filtered, vec![col("id"), col("first_name")]);

        assert_eq!(
            format_plan(&plan),
            "Projection: #id, #first_name\n\
             \tSelection: #state = 'CO'\n\
             \t\tScan: employee; projection=None\n"
        );
    }
}
