//! Metadata filtering expressions
//!
//! ## Table of Contents
//! - **FilterExpr**: Composable filter expressions (AND/OR/NOT/comparisons)
//! - **FilterOp**: Comparison operators (eq, neq, gt, gte, lt, lte, in, contains)
//! - **matches**: Evaluate filter against metadata

use crate::metadata::Metadata;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Filter expression for metadata-based filtering
///
/// Supports composable boolean logic (AND, OR, NOT) and various comparison
/// operators for flexible query filtering.
///
/// # Example
/// ```rust
/// use embedvec::FilterExpr;
///
/// let filter = FilterExpr::eq("category", "finance")
///     .and(FilterExpr::gt("timestamp", 1730000000))
///     .and(FilterExpr::not(FilterExpr::eq("status", "archived")));
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FilterExpr {
    /// Logical AND of two expressions
    And(Box<FilterExpr>, Box<FilterExpr>),

    /// Logical OR of two expressions
    Or(Box<FilterExpr>, Box<FilterExpr>),

    /// Logical NOT of an expression
    Not(Box<FilterExpr>),

    /// Comparison operation on a field
    Compare {
        /// Field name to compare
        field: String,
        /// Comparison operator
        op: FilterOp,
        /// Value to compare against
        value: Value,
    },

    /// Check if field exists
    Exists(String),

    /// Always true (no filter)
    All,

    /// Always false (match nothing)
    None,
}

/// Comparison operators for filter expressions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FilterOp {
    /// Equal
    Eq,
    /// Not equal
    Neq,
    /// Greater than
    Gt,
    /// Greater than or equal
    Gte,
    /// Less than
    Lt,
    /// Less than or equal
    Lte,
    /// Value is in array
    In,
    /// String contains substring
    Contains,
    /// String starts with prefix
    StartsWith,
    /// String ends with suffix
    EndsWith,
}

impl FilterExpr {
    /// Create an equality filter: field == value
    pub fn eq(field: impl Into<String>, value: impl Into<Value>) -> Self {
        FilterExpr::Compare {
            field: field.into(),
            op: FilterOp::Eq,
            value: value.into(),
        }
    }

    /// Create a not-equal filter: field != value
    pub fn neq(field: impl Into<String>, value: impl Into<Value>) -> Self {
        FilterExpr::Compare {
            field: field.into(),
            op: FilterOp::Neq,
            value: value.into(),
        }
    }

    /// Create a greater-than filter: field > value
    pub fn gt(field: impl Into<String>, value: impl Into<Value>) -> Self {
        FilterExpr::Compare {
            field: field.into(),
            op: FilterOp::Gt,
            value: value.into(),
        }
    }

    /// Create a greater-than-or-equal filter: field >= value
    pub fn gte(field: impl Into<String>, value: impl Into<Value>) -> Self {
        FilterExpr::Compare {
            field: field.into(),
            op: FilterOp::Gte,
            value: value.into(),
        }
    }

    /// Create a less-than filter: field < value
    pub fn lt(field: impl Into<String>, value: impl Into<Value>) -> Self {
        FilterExpr::Compare {
            field: field.into(),
            op: FilterOp::Lt,
            value: value.into(),
        }
    }

    /// Create a less-than-or-equal filter: field <= value
    pub fn lte(field: impl Into<String>, value: impl Into<Value>) -> Self {
        FilterExpr::Compare {
            field: field.into(),
            op: FilterOp::Lte,
            value: value.into(),
        }
    }

    /// Create an "in" filter: field in [values...]
    pub fn in_values(field: impl Into<String>, values: Vec<Value>) -> Self {
        FilterExpr::Compare {
            field: field.into(),
            op: FilterOp::In,
            value: Value::Array(values),
        }
    }

    /// Create a contains filter: field contains substring
    pub fn contains(field: impl Into<String>, substring: impl Into<String>) -> Self {
        FilterExpr::Compare {
            field: field.into(),
            op: FilterOp::Contains,
            value: Value::String(substring.into()),
        }
    }

    /// Create a starts-with filter
    pub fn starts_with(field: impl Into<String>, prefix: impl Into<String>) -> Self {
        FilterExpr::Compare {
            field: field.into(),
            op: FilterOp::StartsWith,
            value: Value::String(prefix.into()),
        }
    }

    /// Create an ends-with filter
    pub fn ends_with(field: impl Into<String>, suffix: impl Into<String>) -> Self {
        FilterExpr::Compare {
            field: field.into(),
            op: FilterOp::EndsWith,
            value: Value::String(suffix.into()),
        }
    }

    /// Create an exists filter: field exists
    pub fn exists(field: impl Into<String>) -> Self {
        FilterExpr::Exists(field.into())
    }

    /// Create a NOT expression
    pub fn not(expr: FilterExpr) -> Self {
        FilterExpr::Not(Box::new(expr))
    }

    /// Combine with AND
    pub fn and(self, other: FilterExpr) -> Self {
        FilterExpr::And(Box::new(self), Box::new(other))
    }

    /// Combine with OR
    pub fn or(self, other: FilterExpr) -> Self {
        FilterExpr::Or(Box::new(self), Box::new(other))
    }

    /// Evaluate filter against metadata
    ///
    /// # Arguments
    /// * `metadata` - The metadata to evaluate against
    ///
    /// # Returns
    /// true if metadata matches the filter
    pub fn matches(&self, metadata: &Metadata) -> bool {
        match self {
            FilterExpr::All => true,
            FilterExpr::None => false,

            FilterExpr::And(a, b) => a.matches(metadata) && b.matches(metadata),
            FilterExpr::Or(a, b) => a.matches(metadata) || b.matches(metadata),
            FilterExpr::Not(expr) => !expr.matches(metadata),

            FilterExpr::Exists(field) => metadata.get(field).is_some(),

            FilterExpr::Compare { field, op, value } => {
                if let Some(field_value) = metadata.get(field) {
                    compare_values(field_value, *op, value)
                } else {
                    false
                }
            }
        }
    }
}

/// Compare two JSON values using the specified operator
fn compare_values(field_value: &Value, op: FilterOp, filter_value: &Value) -> bool {
    match op {
        FilterOp::Eq => field_value == filter_value,
        FilterOp::Neq => field_value != filter_value,

        FilterOp::Gt => compare_numeric(field_value, filter_value, |a, b| a > b),
        FilterOp::Gte => compare_numeric(field_value, filter_value, |a, b| a >= b),
        FilterOp::Lt => compare_numeric(field_value, filter_value, |a, b| a < b),
        FilterOp::Lte => compare_numeric(field_value, filter_value, |a, b| a <= b),

        FilterOp::In => {
            if let Value::Array(arr) = filter_value {
                arr.contains(field_value)
            } else {
                false
            }
        }

        FilterOp::Contains => {
            if let (Value::String(field_str), Value::String(filter_str)) =
                (field_value, filter_value)
            {
                field_str.contains(filter_str.as_str())
            } else {
                false
            }
        }

        FilterOp::StartsWith => {
            if let (Value::String(field_str), Value::String(filter_str)) =
                (field_value, filter_value)
            {
                field_str.starts_with(filter_str.as_str())
            } else {
                false
            }
        }

        FilterOp::EndsWith => {
            if let (Value::String(field_str), Value::String(filter_str)) =
                (field_value, filter_value)
            {
                field_str.ends_with(filter_str.as_str())
            } else {
                false
            }
        }
    }
}

/// Compare numeric values with a comparison function
fn compare_numeric<F>(field_value: &Value, filter_value: &Value, cmp: F) -> bool
where
    F: Fn(f64, f64) -> bool,
{
    let field_num = value_to_f64(field_value);
    let filter_num = value_to_f64(filter_value);

    match (field_num, filter_num) {
        (Some(a), Some(b)) => cmp(a, b),
        _ => false,
    }
}

/// Convert JSON value to f64 if possible
fn value_to_f64(value: &Value) -> Option<f64> {
    match value {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => s.parse().ok(),
        _ => None,
    }
}

/// Parse a simple filter from a JSON object (Python-friendly shorthand)
///
/// # Example
/// ```rust
/// use embedvec::filter::parse_simple_filter;
///
/// let filter = parse_simple_filter(&serde_json::json!({"category": "news"}));
/// ```
pub fn parse_simple_filter(obj: &Value) -> Option<FilterExpr> {
    if let Value::Object(map) = obj {
        let mut expr: Option<FilterExpr> = None;

        for (key, value) in map {
            let field_filter = FilterExpr::eq(key.clone(), value.clone());
            expr = Some(match expr {
                Some(e) => e.and(field_filter),
                None => field_filter,
            });
        }

        expr
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_eq_filter() {
        let meta = json!({"category": "finance", "count": 42});
        let filter = FilterExpr::eq("category", "finance");
        assert!(filter.matches(&meta));

        let filter2 = FilterExpr::eq("category", "tech");
        assert!(!filter2.matches(&meta));
    }

    #[test]
    fn test_numeric_comparison() {
        let meta = json!({"count": 42, "score": 3.14});

        assert!(FilterExpr::gt("count", 40).matches(&meta));
        assert!(FilterExpr::lt("count", 50).matches(&meta));
        assert!(FilterExpr::gte("count", 42).matches(&meta));
        assert!(FilterExpr::lte("count", 42).matches(&meta));
        assert!(!FilterExpr::gt("count", 42).matches(&meta));
    }

    #[test]
    fn test_and_or_not() {
        let meta = json!({"a": 1, "b": 2});

        let filter = FilterExpr::eq("a", 1).and(FilterExpr::eq("b", 2));
        assert!(filter.matches(&meta));

        let filter2 = FilterExpr::eq("a", 1).and(FilterExpr::eq("b", 3));
        assert!(!filter2.matches(&meta));

        let filter3 = FilterExpr::eq("a", 1).or(FilterExpr::eq("b", 3));
        assert!(filter3.matches(&meta));

        let filter4 = FilterExpr::not(FilterExpr::eq("a", 2));
        assert!(filter4.matches(&meta));
    }

    #[test]
    fn test_in_filter() {
        let meta = json!({"status": "active"});
        let filter = FilterExpr::in_values("status", vec![json!("active"), json!("pending")]);
        assert!(filter.matches(&meta));

        let filter2 = FilterExpr::in_values("status", vec![json!("archived"), json!("deleted")]);
        assert!(!filter2.matches(&meta));
    }

    #[test]
    fn test_string_filters() {
        let meta = json!({"name": "hello_world_test"});

        assert!(FilterExpr::contains("name", "world").matches(&meta));
        assert!(FilterExpr::starts_with("name", "hello").matches(&meta));
        assert!(FilterExpr::ends_with("name", "test").matches(&meta));
        assert!(!FilterExpr::contains("name", "foo").matches(&meta));
    }

    #[test]
    fn test_exists_filter() {
        let meta = json!({"present": 1});

        assert!(FilterExpr::exists("present").matches(&meta));
        assert!(!FilterExpr::exists("missing").matches(&meta));
    }

    #[test]
    fn test_simple_filter_parse() {
        let meta = json!({"category": "news", "status": "active"});
        let filter_obj = json!({"category": "news"});

        let filter = parse_simple_filter(&filter_obj).unwrap();
        assert!(filter.matches(&meta));
    }
}
