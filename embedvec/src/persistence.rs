//! Persistence Backend Module
//!
//! ## Table of Contents
//! - **PersistenceBackend**: Trait for pluggable storage backends
//! - **SledBackend**: Sled-based persistence (default, pure Rust)
//! - **RocksDbBackend**: RocksDB-based persistence (higher performance)
//! - **BackendConfig**: Configuration for backend selection

use crate::error::{EmbedVecError, Result};
use serde::{de::DeserializeOwned, Serialize};

/// Persistence backend type selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BackendType {
    /// Sled - Pure Rust embedded database (default)
    #[default]
    Sled,
    /// RocksDB - High-performance LSM-tree database
    RocksDb,
}

/// Configuration for persistence backend
#[derive(Debug, Clone)]
pub struct BackendConfig {
    /// Path to the database directory
    pub path: String,
    /// Backend type to use
    pub backend_type: BackendType,
    /// Whether to create the database if it doesn't exist
    pub create_if_missing: bool,
    /// Cache size in bytes (for RocksDB)
    pub cache_size: Option<usize>,
}

impl BackendConfig {
    /// Create a new backend config with default settings
    pub fn new(path: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            backend_type: BackendType::Sled,
            create_if_missing: true,
            cache_size: None,
        }
    }

    /// Set the backend type
    pub fn backend(mut self, backend_type: BackendType) -> Self {
        self.backend_type = backend_type;
        self
    }

    /// Set cache size (primarily for RocksDB)
    pub fn cache_size(mut self, size: usize) -> Self {
        self.cache_size = Some(size);
        self
    }
}

/// Trait for persistence backends
/// 
/// Provides a common interface for different storage engines like sled and rocksdb.
pub trait PersistenceBackend: Send + Sync {
    /// Get a value by key
    fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>>;
    
    /// Set a value for a key
    fn set(&self, key: &[u8], value: &[u8]) -> Result<()>;
    
    /// Delete a key
    fn delete(&self, key: &[u8]) -> Result<()>;
    
    /// Check if a key exists
    fn contains(&self, key: &[u8]) -> Result<bool>;
    
    /// Iterate over all key-value pairs with a prefix
    fn scan_prefix(&self, prefix: &[u8]) -> Result<Vec<(Vec<u8>, Vec<u8>)>>;
    
    /// Flush all pending writes to disk
    fn flush(&self) -> Result<()>;
    
    /// Get the backend type
    fn backend_type(&self) -> BackendType;
}

/// Extension trait for typed get/set operations
pub trait PersistenceBackendExt: PersistenceBackend {
    /// Get a typed value by key
    fn get_typed<T: DeserializeOwned>(&self, key: &[u8]) -> Result<Option<T>> {
        match self.get(key)? {
            Some(bytes) => {
                let value: T = serde_json::from_slice(&bytes)
                    .map_err(|e| EmbedVecError::SerializationError(e.to_string()))?;
                Ok(Some(value))
            }
            None => Ok(None),
        }
    }
    
    /// Set a typed value for a key
    fn set_typed<T: Serialize>(&self, key: &[u8], value: &T) -> Result<()> {
        let bytes = serde_json::to_vec(value)
            .map_err(|e| EmbedVecError::SerializationError(e.to_string()))?;
        self.set(key, &bytes)
    }
}

// Blanket implementation for all PersistenceBackend types
impl<T: PersistenceBackend + ?Sized> PersistenceBackendExt for T {}

// =============================================================================
// Sled Backend Implementation
// =============================================================================

#[cfg(feature = "persistence-sled")]
mod sled_backend {
    use super::*;
    
    /// Sled-based persistence backend
    /// 
    /// Pure Rust embedded database with ACID guarantees.
    /// Good default choice for most use cases.
    pub struct SledBackend {
        db: sled::Db,
    }
    
    impl SledBackend {
        /// Open or create a sled database at the given path
        pub fn open(config: &BackendConfig) -> Result<Self> {
            let db = sled::open(&config.path)
                .map_err(|e| EmbedVecError::PersistenceError(e.to_string()))?;
            Ok(Self { db })
        }
    }
    
    impl PersistenceBackend for SledBackend {
        fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
            self.db
                .get(key)
                .map(|opt| opt.map(|ivec| ivec.to_vec()))
                .map_err(|e| EmbedVecError::PersistenceError(e.to_string()))
        }
        
        fn set(&self, key: &[u8], value: &[u8]) -> Result<()> {
            self.db
                .insert(key, value)
                .map(|_| ())
                .map_err(|e| EmbedVecError::PersistenceError(e.to_string()))
        }
        
        fn delete(&self, key: &[u8]) -> Result<()> {
            self.db
                .remove(key)
                .map(|_| ())
                .map_err(|e| EmbedVecError::PersistenceError(e.to_string()))
        }
        
        fn contains(&self, key: &[u8]) -> Result<bool> {
            self.db
                .contains_key(key)
                .map_err(|e| EmbedVecError::PersistenceError(e.to_string()))
        }
        
        fn scan_prefix(&self, prefix: &[u8]) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
            let mut results = Vec::new();
            for item in self.db.scan_prefix(prefix) {
                let (key, value) = item
                    .map_err(|e| EmbedVecError::PersistenceError(e.to_string()))?;
                results.push((key.to_vec(), value.to_vec()));
            }
            Ok(results)
        }
        
        fn flush(&self) -> Result<()> {
            self.db
                .flush()
                .map(|_| ())
                .map_err(|e| EmbedVecError::PersistenceError(e.to_string()))
        }
        
        fn backend_type(&self) -> BackendType {
            BackendType::Sled
        }
    }
}

#[cfg(feature = "persistence-sled")]
pub use sled_backend::SledBackend;

// =============================================================================
// RocksDB Backend Implementation
// =============================================================================

#[cfg(feature = "persistence-rocksdb")]
mod rocksdb_backend {
    use super::*;
    use rocksdb::{DB, Options, IteratorMode};
    
    /// RocksDB-based persistence backend
    /// 
    /// High-performance LSM-tree database from Facebook.
    /// Better for write-heavy workloads and large datasets.
    pub struct RocksDbBackend {
        db: DB,
    }
    
    impl RocksDbBackend {
        /// Open or create a RocksDB database at the given path
        pub fn open(config: &BackendConfig) -> Result<Self> {
            let mut opts = Options::default();
            opts.create_if_missing(config.create_if_missing);
            
            // Set cache size if specified
            if let Some(cache_size) = config.cache_size {
                opts.set_write_buffer_size(cache_size / 4);
                // Note: Block cache requires more setup with BlockBasedOptions
            }
            
            // Optimize for point lookups
            opts.set_max_open_files(256);
            opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
            
            let db = DB::open(&opts, &config.path)
                .map_err(|e| EmbedVecError::PersistenceError(e.to_string()))?;
            
            Ok(Self { db })
        }
    }
    
    impl PersistenceBackend for RocksDbBackend {
        fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
            self.db
                .get(key)
                .map_err(|e| EmbedVecError::PersistenceError(e.to_string()))
        }
        
        fn set(&self, key: &[u8], value: &[u8]) -> Result<()> {
            self.db
                .put(key, value)
                .map_err(|e| EmbedVecError::PersistenceError(e.to_string()))
        }
        
        fn delete(&self, key: &[u8]) -> Result<()> {
            self.db
                .delete(key)
                .map_err(|e| EmbedVecError::PersistenceError(e.to_string()))
        }
        
        fn contains(&self, key: &[u8]) -> Result<bool> {
            self.db
                .get(key)
                .map(|opt| opt.is_some())
                .map_err(|e| EmbedVecError::PersistenceError(e.to_string()))
        }
        
        fn scan_prefix(&self, prefix: &[u8]) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
            let mut results = Vec::new();
            let iter = self.db.iterator(IteratorMode::From(prefix, rocksdb::Direction::Forward));
            
            for item in iter {
                let (key, value) = item
                    .map_err(|e| EmbedVecError::PersistenceError(e.to_string()))?;
                
                // Stop if we've passed the prefix
                if !key.starts_with(prefix) {
                    break;
                }
                
                results.push((key.to_vec(), value.to_vec()));
            }
            
            Ok(results)
        }
        
        fn flush(&self) -> Result<()> {
            self.db
                .flush()
                .map_err(|e| EmbedVecError::PersistenceError(e.to_string()))
        }
        
        fn backend_type(&self) -> BackendType {
            BackendType::RocksDb
        }
    }
}

#[cfg(feature = "persistence-rocksdb")]
pub use rocksdb_backend::RocksDbBackend;

// =============================================================================
// Backend Factory
// =============================================================================

/// Create a persistence backend based on configuration
#[cfg(any(feature = "persistence-sled", feature = "persistence-rocksdb"))]
pub fn create_backend(config: &BackendConfig) -> Result<Box<dyn PersistenceBackend>> {
    match config.backend_type {
        #[cfg(feature = "persistence-sled")]
        BackendType::Sled => {
            let backend = SledBackend::open(config)?;
            Ok(Box::new(backend))
        }
        #[cfg(feature = "persistence-rocksdb")]
        BackendType::RocksDb => {
            let backend = RocksDbBackend::open(config)?;
            Ok(Box::new(backend))
        }
        #[cfg(not(feature = "persistence-sled"))]
        BackendType::Sled => {
            Err(EmbedVecError::PersistenceError(
                "Sled backend not enabled. Enable 'persistence-sled' feature.".to_string()
            ))
        }
        #[cfg(not(feature = "persistence-rocksdb"))]
        BackendType::RocksDb => {
            Err(EmbedVecError::PersistenceError(
                "RocksDB backend not enabled. Enable 'persistence-rocksdb' feature.".to_string()
            ))
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_backend_config() {
        let config = BackendConfig::new("/tmp/test")
            .backend(BackendType::RocksDb)
            .cache_size(64 * 1024 * 1024);
        
        assert_eq!(config.backend_type, BackendType::RocksDb);
        assert_eq!(config.cache_size, Some(64 * 1024 * 1024));
    }
    
    #[cfg(feature = "persistence-sled")]
    #[test]
    fn test_sled_backend() {
        use std::env::temp_dir;
        use std::time::{SystemTime, UNIX_EPOCH};
        
        // Create unique temp path
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = temp_dir().join(format!("embedvec_test_{}", timestamp));
        let config = BackendConfig::new(path.to_str().unwrap());
        
        let backend = SledBackend::open(&config).unwrap();
        
        // Test set/get
        backend.set(b"key1", b"value1").unwrap();
        let value = backend.get(b"key1").unwrap();
        assert_eq!(value, Some(b"value1".to_vec()));
        
        // Test contains
        assert!(backend.contains(b"key1").unwrap());
        assert!(!backend.contains(b"key2").unwrap());
        
        // Test delete
        backend.delete(b"key1").unwrap();
        assert!(!backend.contains(b"key1").unwrap());
        
        // Cleanup
        drop(backend);
        let _ = std::fs::remove_dir_all(&path);
    }
}
