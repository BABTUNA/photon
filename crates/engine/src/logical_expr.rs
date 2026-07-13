//! Logical expressions (book: "Logical Plans").
//!
//! Expressions are trees evaluated against the rows a plan node produces —
//! `#salary > 4000` is a comparison over a column reference and a literal.
//! At the logical layer an expression can't compute anything; what it CAN
//! always do is answer "what field (name + type) do I produce against this
//! input?" — that's `to_field`, and it's how Projection and friends (1.15)
//! derive their schemas before any data is read.
//!
//! Deviation from the book: KQuery models expressions as one class per
//! operator; here they're a single enum. The planner (2.8), optimizer (3.5)
//! and SQL frontend (3.10) all become `match` statements over it.

use std::fmt::Display;

use arrow::datatypes::Field;

use crate::datatypes::ScalarValue;
use crate::logical_plan::LogicalPlan;

#[derive(Debug, Clone, PartialEq)]
pub enum LogicalExpr {
    /// A column of the input, referenced by name.
    Column(String),
    /// A constant.
    Literal(ScalarValue),
}

/// Reference a column: `col("salary")`.
pub fn col(name: impl Into<String>) -> LogicalExpr {
    LogicalExpr::Column(name.into())
}

/// A constant: `lit(4)`, `lit(1.5)`, `lit("CO")`.
pub fn lit(value: impl Into<ScalarValue>) -> LogicalExpr {
    LogicalExpr::Literal(value.into())
}

impl LogicalExpr {
    /// The field this expression produces when evaluated against `input`'s
    /// rows. Columns resolve (and type-check) against the input schema;
    /// literals carry their own type and are named by their value.
    pub fn to_field(&self, input: &dyn LogicalPlan) -> Field {
        match self {
            LogicalExpr::Column(name) => input
                .schema()
                .field_with_name(name)
                .unwrap_or_else(|_| panic!("no column named {name:?} in input schema"))
                .clone(),
            LogicalExpr::Literal(value) => {
                let name = match value {
                    ScalarValue::Utf8(s) => s.clone(),
                    other => other.to_string(),
                };
                Field::new(name, value.data_type(), true)
            }
        }
    }
}

impl Display for LogicalExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LogicalExpr::Column(name) => write!(f, "#{name}"),
            LogicalExpr::Literal(value) => write!(f, "{value}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datasource::CsvDataSource;
    use crate::logical_plan::Scan;
    use arrow::datatypes::DataType;
    use std::sync::Arc;

    fn employee_scan() -> Scan {
        let source = Arc::new(CsvDataSource::new("testdata/employee.csv", None, 1024));
        Scan::new("employee", source, vec![])
    }

    #[test]
    fn column_resolves_name_and_type_from_input() {
        let field = col("salary").to_field(&employee_scan());
        assert_eq!(field.name(), "salary");
        assert_eq!(field.data_type(), &DataType::Int64);
    }

    #[test]
    #[should_panic(expected = "no column named")]
    fn unknown_column_panics() {
        col("bogus").to_field(&employee_scan());
    }

    #[test]
    fn literals_carry_their_own_type() {
        let scan = employee_scan();

        let int = lit(4).to_field(&scan);
        assert_eq!(int.name(), "4");
        assert_eq!(int.data_type(), &DataType::Int32);

        let text = lit("CO").to_field(&scan);
        assert_eq!(text.name(), "CO");
        assert_eq!(text.data_type(), &DataType::Utf8);
    }

    #[test]
    fn display_matches_book_notation() {
        assert_eq!(col("id").to_string(), "#id");
        assert_eq!(lit(4i64).to_string(), "4");
        assert_eq!(lit("CO").to_string(), "'CO'");
    }
}
