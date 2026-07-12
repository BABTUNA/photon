//! Logical plans (book: "Logical Plans").
//!
//! A logical plan is a tree describing WHAT a query computes — which data,
//! which columns, which filters — with no opinion on HOW. The how (hash vs
//! sort, serial vs parallel) is the physical plan's job (commit 2.3). Every
//! frontend (DataFrame in 2.1, SQL in 3.10) produces this same tree.

use std::fmt::Display;
use std::sync::Arc;

use arrow::datatypes::SchemaRef;

use crate::datasource::DataSource;
use crate::datatypes::project_by_name;

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
}
