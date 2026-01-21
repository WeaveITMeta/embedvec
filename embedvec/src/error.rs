//! Error types for embedvec
//!
//! ## Table of Contents
//! - **EmbedVecError**: Main error enum covering all failure modes
//! - **Result**: Type alias for Result<T, EmbedVecError>

use thiserror::Error;

/// Main error type for embedvec operations
#[derive(Error, Debug)]
pub enum EmbedVecError {
    /// Vector dimension is invalid (zero)
    #[error("Invalid dimension: {0}. Dimension must be > 0")]
    InvalidDimension(usize),

    /// Vector dimension doesn't match database configuration
    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch {
        /// Expected dimension
        expected: usize,
        /// Actual dimension provided
        got: usize,
    },

    /// Number of vectors and payloads don't match in batch operation
    #[error("Mismatched lengths: {vectors} vectors, {payloads} payloads")]
    MismatchedLengths {
        /// Number of vectors
        vectors: usize,
        /// Number of payloads
        payloads: usize,
    },

    /// Vector ID not found in storage
    #[error("Vector ID {0} not found")]
    VectorNotFound(usize),

    /// Index is empty, cannot perform search
    #[error("Index is empty")]
    EmptyIndex,

    /// Persistence operation failed
    #[error("Persistence error: {0}")]
    PersistenceError(String),

    /// Serialization/deserialization error
    #[error("Serialization error: {0}")]
    SerializationError(String),

    /// Quantization error
    #[error("Quantization error: {0}")]
    QuantizationError(String),

    /// Invalid parameter value
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    /// Internal error (should not happen in normal operation)
    #[error("Internal error: {0}")]
    InternalError(String),
}

/// Result type alias for embedvec operations
pub type Result<T> = std::result::Result<T, EmbedVecError>;

impl From<serde_json::Error> for EmbedVecError {
    fn from(e: serde_json::Error) -> Self {
        EmbedVecError::SerializationError(e.to_string())
    }
}

#[cfg(feature = "persistence")]
impl From<sled::Error> for EmbedVecError {
    fn from(e: sled::Error) -> Self {
        EmbedVecError::PersistenceError(e.to_string())
    }
}
