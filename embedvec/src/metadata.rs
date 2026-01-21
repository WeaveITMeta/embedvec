//! Metadata storage and manipulation
//!
//! ## Table of Contents
//! - **Metadata**: Type alias for JSON-compatible metadata
//! - **MetadataExt**: Extension trait for convenient field access
//! - **Helper functions**: Conversion utilities

use serde_json::Value;

/// Metadata type for vector payloads
///
/// Uses serde_json::Value for flexible JSON-compatible storage.
/// Supports arbitrary nested structures, arrays, and primitive types.
///
/// # Example
/// ```rust
/// use embedvec::Metadata;
///
/// let meta: Metadata = serde_json::json!({
///     "doc_id": "123",
///     "category": "finance",
///     "timestamp": 1737400000,
///     "tags": ["important", "reviewed"]
/// });
/// ```
pub type Metadata = Value;

/// Extension trait for Metadata operations
pub trait MetadataExt {
    /// Get a string field from metadata
    fn get_str(&self, key: &str) -> Option<&str>;

    /// Get an integer field from metadata
    fn get_i64(&self, key: &str) -> Option<i64>;

    /// Get a float field from metadata
    fn get_f64(&self, key: &str) -> Option<f64>;

    /// Get a boolean field from metadata
    fn get_bool(&self, key: &str) -> Option<bool>;

    /// Check if metadata has a specific key
    fn has_key(&self, key: &str) -> bool;
}

impl MetadataExt for Metadata {
    fn get_str(&self, key: &str) -> Option<&str> {
        self.get(key)?.as_str()
    }

    fn get_i64(&self, key: &str) -> Option<i64> {
        self.get(key)?.as_i64()
    }

    fn get_f64(&self, key: &str) -> Option<f64> {
        self.get(key)?.as_f64()
    }

    fn get_bool(&self, key: &str) -> Option<bool> {
        self.get(key)?.as_bool()
    }

    fn has_key(&self, key: &str) -> bool {
        self.get(key).is_some()
    }
}

/// Create metadata from a key-value pair
pub fn metadata_from_kv(key: &str, value: &str) -> Metadata {
    serde_json::json!({ key: value })
}

/// Create metadata from a HashMap
pub fn metadata_from_hashmap(map: std::collections::HashMap<String, String>) -> Metadata {
    serde_json::to_value(map).unwrap_or(Value::Null)
}

/// Create metadata from a serde_json::Map
pub fn metadata_from_map(map: serde_json::Map<String, Value>) -> Metadata {
    Value::Object(map)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metadata_ext() {
        let meta: Metadata = serde_json::json!({
            "name": "test",
            "count": 42,
            "score": 3.14,
            "active": true
        });

        assert_eq!(meta.get_str("name"), Some("test"));
        assert_eq!(meta.get_i64("count"), Some(42));
        assert!((meta.get_f64("score").unwrap() - 3.14).abs() < 1e-6);
        assert_eq!(meta.get_bool("active"), Some(true));
        assert!(meta.has_key("name"));
        assert!(!meta.has_key("missing"));
    }
}
