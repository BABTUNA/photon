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
    /// Two sub-expressions combined by an operator: `#a > 4`, `#x AND #y`.
    BinaryExpr {
        left: Box<LogicalExpr>,
        op: Operator,
        right: Box<LogicalExpr>,
    },
}

/// The operator taxonomy: three families with different result types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Operator {
    // comparisons — result is Boolean, operands any comparable type
    Eq,
    Neq,
    Gt,
    GtEq,
    Lt,
    LtEq,
    // boolean algebra — Boolean in, Boolean out
    And,
    Or,
    // math — result keeps the (left) operand's type
    Add,
    Subtract,
    Multiply,
    Divide,
    Modulus,
}

impl Operator {
    /// Short name, used as the output field name (book convention).
    pub fn name(&self) -> &'static str {
        match self {
            Operator::Eq => "eq",
            Operator::Neq => "neq",
            Operator::Gt => "gt",
            Operator::GtEq => "gteq",
            Operator::Lt => "lt",
            Operator::LtEq => "lteq",
            Operator::And => "and",
            Operator::Or => "or",
            Operator::Add => "add",
            Operator::Subtract => "subtract",
            Operator::Multiply => "mult",
            Operator::Divide => "div",
            Operator::Modulus => "mod",
        }
    }

    /// Comparisons and boolean algebra produce Boolean; math does not.
    pub fn produces_boolean(&self) -> bool {
        !matches!(
            self,
            Operator::Add
                | Operator::Subtract
                | Operator::Multiply
                | Operator::Divide
                | Operator::Modulus
        )
    }
}

impl Display for Operator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let symbol = match self {
            Operator::Eq => "=",
            Operator::Neq => "!=",
            Operator::Gt => ">",
            Operator::GtEq => ">=",
            Operator::Lt => "<",
            Operator::LtEq => "<=",
            Operator::And => "AND",
            Operator::Or => "OR",
            Operator::Add => "+",
            Operator::Subtract => "-",
            Operator::Multiply => "*",
            Operator::Divide => "/",
            Operator::Modulus => "%",
        };
        write!(f, "{symbol}")
    }
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
            LogicalExpr::BinaryExpr { left, op, .. } => {
                if op.produces_boolean() {
                    Field::new(op.name(), arrow::datatypes::DataType::Boolean, true)
                } else {
                    // Book convention: a math expression's type follows its
                    // left operand. Real coercion happens physically (2.5).
                    Field::new(op.name(), left.to_field(input).data_type().clone(), true)
                }
            }
        }
    }

    fn binary(self, op: Operator, rhs: LogicalExpr) -> LogicalExpr {
        LogicalExpr::BinaryExpr {
            left: Box::new(self),
            op,
            right: Box::new(rhs),
        }
    }
}

/// Fluent builders: `col("salary").gt(lit(4000)).and(col("state").eq(lit("CO")))`.
///
/// Some names shadow std traits (`eq`, `add`, ...) — intentional, these are
/// expression-tree constructors, not implementations of those traits.
#[allow(clippy::should_implement_trait)]
impl LogicalExpr {
    pub fn eq(self, rhs: LogicalExpr) -> LogicalExpr {
        self.binary(Operator::Eq, rhs)
    }
    pub fn neq(self, rhs: LogicalExpr) -> LogicalExpr {
        self.binary(Operator::Neq, rhs)
    }
    pub fn gt(self, rhs: LogicalExpr) -> LogicalExpr {
        self.binary(Operator::Gt, rhs)
    }
    pub fn gt_eq(self, rhs: LogicalExpr) -> LogicalExpr {
        self.binary(Operator::GtEq, rhs)
    }
    pub fn lt(self, rhs: LogicalExpr) -> LogicalExpr {
        self.binary(Operator::Lt, rhs)
    }
    pub fn lt_eq(self, rhs: LogicalExpr) -> LogicalExpr {
        self.binary(Operator::LtEq, rhs)
    }
    pub fn and(self, rhs: LogicalExpr) -> LogicalExpr {
        self.binary(Operator::And, rhs)
    }
    pub fn or(self, rhs: LogicalExpr) -> LogicalExpr {
        self.binary(Operator::Or, rhs)
    }
    pub fn add(self, rhs: LogicalExpr) -> LogicalExpr {
        self.binary(Operator::Add, rhs)
    }
    pub fn subtract(self, rhs: LogicalExpr) -> LogicalExpr {
        self.binary(Operator::Subtract, rhs)
    }
    pub fn multiply(self, rhs: LogicalExpr) -> LogicalExpr {
        self.binary(Operator::Multiply, rhs)
    }
    pub fn divide(self, rhs: LogicalExpr) -> LogicalExpr {
        self.binary(Operator::Divide, rhs)
    }
    pub fn modulus(self, rhs: LogicalExpr) -> LogicalExpr {
        self.binary(Operator::Modulus, rhs)
    }
}

impl Display for LogicalExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LogicalExpr::Column(name) => write!(f, "#{name}"),
            LogicalExpr::Literal(value) => write!(f, "{value}"),
            LogicalExpr::BinaryExpr { left, op, right } => write!(f, "{left} {op} {right}"),
        }
    }
}

/// Which aggregate function.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggregateFunc {
    Sum,
    Min,
    Max,
    Avg,
    Count,
}

impl AggregateFunc {
    pub fn name(&self) -> &'static str {
        match self {
            AggregateFunc::Sum => "SUM",
            AggregateFunc::Min => "MIN",
            AggregateFunc::Max => "MAX",
            AggregateFunc::Avg => "AVG",
            AggregateFunc::Count => "COUNT",
        }
    }
}

/// An aggregate over an input expression: `SUM(#salary)`.
///
/// Deliberately NOT a `LogicalExpr` variant: aggregates collapse many rows
/// into one, so they only make sense inside an Aggregate plan node (1.15).
/// Keeping them a separate type makes "aggregates only appear in Aggregate
/// nodes" a compile-time guarantee instead of a planner-time check.
#[derive(Debug, Clone, PartialEq)]
pub struct AggregateExpr {
    pub func: AggregateFunc,
    pub expr: LogicalExpr,
}

impl AggregateExpr {
    pub fn to_field(&self, input: &dyn LogicalPlan) -> Field {
        use arrow::datatypes::DataType;
        let data_type = match self.func {
            // COUNT is a row count whatever it counts.
            AggregateFunc::Count => DataType::Int64,
            // AVG of integers is fractional (deviation: the book keeps the
            // operand type, which would truncate at execution time).
            AggregateFunc::Avg => DataType::Float64,
            // SUM/MIN/MAX keep the operand's type.
            _ => self.expr.to_field(input).data_type().clone(),
        };
        Field::new(self.func.name(), data_type, true)
    }
}

impl Display for AggregateExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}({})", self.func.name(), self.expr)
    }
}

/// `sum(col("salary"))`, `count(col("id"))`, ...
pub fn sum(expr: LogicalExpr) -> AggregateExpr {
    AggregateExpr {
        func: AggregateFunc::Sum,
        expr,
    }
}
pub fn min(expr: LogicalExpr) -> AggregateExpr {
    AggregateExpr {
        func: AggregateFunc::Min,
        expr,
    }
}
pub fn max(expr: LogicalExpr) -> AggregateExpr {
    AggregateExpr {
        func: AggregateFunc::Max,
        expr,
    }
}
pub fn avg(expr: LogicalExpr) -> AggregateExpr {
    AggregateExpr {
        func: AggregateFunc::Avg,
        expr,
    }
}
pub fn count(expr: LogicalExpr) -> AggregateExpr {
    AggregateExpr {
        func: AggregateFunc::Count,
        expr,
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

    #[test]
    fn comparisons_produce_boolean_fields() {
        let expr = col("salary").gt(lit(4000i64));
        let field = expr.to_field(&employee_scan());
        assert_eq!(field.name(), "gt");
        assert_eq!(field.data_type(), &DataType::Boolean);
    }

    #[test]
    fn math_keeps_the_left_operand_type() {
        let expr = col("salary").multiply(lit(2i64));
        let field = expr.to_field(&employee_scan());
        assert_eq!(field.name(), "mult");
        assert_eq!(field.data_type(), &DataType::Int64);
    }

    #[test]
    fn builders_nest_and_display_readably() {
        let expr = col("state")
            .eq(lit("CO"))
            .and(col("salary").gt_eq(lit(10000i64)));
        assert_eq!(expr.to_string(), "#state = 'CO' AND #salary >= 10000");
    }

    #[test]
    fn boolean_algebra_type_checks_over_comparisons() {
        let expr = col("salary").lt(lit(1i64)).or(col("salary").gt(lit(2i64)));
        assert_eq!(
            expr.to_field(&employee_scan()).data_type(),
            &DataType::Boolean
        );
    }

    #[test]
    fn aggregate_field_types_follow_the_function() {
        let scan = employee_scan();

        let s = sum(col("salary")).to_field(&scan);
        assert_eq!((s.name().as_str(), s.data_type()), ("SUM", &DataType::Int64));

        let c = count(col("state")).to_field(&scan);
        assert_eq!((c.name().as_str(), c.data_type()), ("COUNT", &DataType::Int64));

        let a = avg(col("salary")).to_field(&scan);
        assert_eq!((a.name().as_str(), a.data_type()), ("AVG", &DataType::Float64));

        let m = min(col("first_name")).to_field(&scan);
        assert_eq!((m.name().as_str(), m.data_type()), ("MIN", &DataType::Utf8));
    }

    #[test]
    fn aggregates_display_like_sql() {
        assert_eq!(sum(col("salary")).to_string(), "SUM(#salary)");
        assert_eq!(
            max(col("salary").multiply(lit(2i64))).to_string(),
            "MAX(#salary * 2)"
        );
    }
}
