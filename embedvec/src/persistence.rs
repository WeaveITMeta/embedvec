//! Persistence Backend Module
//!
//! ## Table of Contents
//! - **PersistenceBackend**: Trait for pluggable storage backends
//! - **SledBackend**: Sled-based persistence (default, pure Rust)
//! - **RocksDbBackend**: RocksDB-based persistence (higher performance)
//! - **PgVectorBackend**: PostgreSQL with pgvector extension (native vector search)
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
    /// PgVector - PostgreSQL with native vector search
    PgVector,
}

/// Configuration for persistence backend
#[derive(Debug, Clone)]
pub struct BackendConfig {
    /// Path to the database directory (or connection string for PgVector)
    pub path: String,
    /// Backend type to use
    pub backend_type: BackendType,
    /// Whether to create the database if it doesn't exist
    pub create_if_missing: bool,
    /// Cache size in bytes (for RocksDB)
    pub cache_size: Option<usize>,
    /// Table name for pgvector (default: "embedvec_vectors")
    pub table_name: Option<String>,
    /// Vector dimension for pgvector
    pub dimension: Option<usize>,
    /// Index type for pgvector: "ivfflat" or "hnsw"
    pub index_type: Option<String>,
}

impl BackendConfig {
    /// Create a new backend config with default settings
    pub fn new(path: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            backend_type: BackendType::Sled,
            create_if_missing: true,
            cache_size: None,
            table_name: None,
            dimension: None,
            index_type: None,
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
    
    /// Set table name for pgvector
    pub fn table_name(mut self, name: impl Into<String>) -> Self {
        self.table_name = Some(name.into());
        self
    }
    
    /// Set vector dimension for pgvector
    pub fn dimension(mut self, dim: usize) -> Self {
        self.dimension = Some(dim);
        self
    }
    
    /// Set index type for pgvector ("ivfflat" or "hnsw")
    pub fn index_type(mut self, idx_type: impl Into<String>) -> Self {
        self.index_type = Some(idx_type.into());
        self
    }
    
    /// Create a pgvector config with connection string
    pub fn pgvector(connection_string: impl Into<String>, dimension: usize) -> Self {
        Self {
            path: connection_string.into(),
            backend_type: BackendType::PgVector,
            create_if_missing: true,
            cache_size: None,
            table_name: Some("embedvec_vectors".to_string()),
            dimension: Some(dimension),
            index_type: Some("hnsw".to_string()),
        }
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
// PgVector Backend Implementation
// =============================================================================

#[cfg(feature = "persistence-pgvector")]
mod pgvector_backend {
    use super::*;
    use sqlx::postgres::PgPoolOptions;
    use sqlx::{PgPool, Row};
    use pgvector::Vector;
    use std::sync::Arc;
    use tokio::sync::RwLock;
    
    /// pgvector-based persistence backend
    /// 
    /// Uses PostgreSQL with the pgvector extension for native vector storage and search.
    /// Supports IVFFlat and HNSW indexes for efficient similarity search.
    pub struct PgVectorBackend {
        pool: PgPool,
        table_name: String,
        dimension: usize,
        /// Cache for metadata (key-value pairs that aren't vectors)
        metadata_cache: Arc<RwLock<std::collections::HashMap<Vec<u8>, Vec<u8>>>>,
    }
    
    impl PgVectorBackend {
        /// Connect to PostgreSQL and initialize the pgvector table
        pub async fn connect(config: &BackendConfig) -> Result<Self> {
            let dimension = config.dimension.ok_or_else(|| {
                EmbedVecError::PersistenceError("Dimension required for pgvector backend".to_string())
            })?;
            
            let table_name = config.table_name.clone()
                .unwrap_or_else(|| "embedvec_vectors".to_string());
            
            let index_type = config.index_type.clone()
                .unwrap_or_else(|| "hnsw".to_string());
            
            // Create connection pool
            let pool = PgPoolOptions::new()
                .max_connections(10)
                .connect(&config.path)
                .await
                .map_err(|e| EmbedVecError::PersistenceError(format!("Failed to connect to PostgreSQL: {}", e)))?;
            
            // Enable pgvector extension
            sqlx::query("CREATE EXTENSION IF NOT EXISTS vector")
                .execute(&pool)
                .await
                .map_err(|e| EmbedVecError::PersistenceError(format!("Failed to create vector extension: {}", e)))?;
            
            // Create vectors table
            let create_table_sql = format!(
                r#"
                CREATE TABLE IF NOT EXISTS {} (
                    id BIGSERIAL PRIMARY KEY,
                    external_id TEXT UNIQUE,
                    embedding vector({}),
                    metadata JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
                "#,
                table_name, dimension
            );
            
            sqlx::query(&create_table_sql)
                .execute(&pool)
                .await
                .map_err(|e| EmbedVecError::PersistenceError(format!("Failed to create table: {}", e)))?;
            
            // Create vector index based on type
            let index_name = format!("{}_embedding_idx", table_name);
            let index_sql = match index_type.as_str() {
                "ivfflat" => format!(
                    "CREATE INDEX IF NOT EXISTS {} ON {} USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)",
                    index_name, table_name
                ),
                "hnsw" | _ => format!(
                    "CREATE INDEX IF NOT EXISTS {} ON {} USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)",
                    index_name, table_name
                ),
            };
            
            sqlx::query(&index_sql)
                .execute(&pool)
                .await
                .map_err(|e| EmbedVecError::PersistenceError(format!("Failed to create index: {}", e)))?;
            
            // Create metadata table for key-value storage
            let metadata_table_sql = format!(
                r#"
                CREATE TABLE IF NOT EXISTS {}_metadata (
                    key BYTEA PRIMARY KEY,
                    value BYTEA
                )
                "#,
                table_name
            );
            
            sqlx::query(&metadata_table_sql)
                .execute(&pool)
                .await
                .map_err(|e| EmbedVecError::PersistenceError(format!("Failed to create metadata table: {}", e)))?;
            
            Ok(Self {
                pool,
                table_name,
                dimension,
                metadata_cache: Arc::new(RwLock::new(std::collections::HashMap::new())),
            })
        }
        
        /// Insert a vector with metadata
        pub async fn insert_vector(
            &self,
            external_id: &str,
            embedding: &[f32],
            metadata: Option<serde_json::Value>,
        ) -> Result<i64> {
            let vector = Vector::from(embedding.to_vec());
            
            let sql = format!(
                r#"
                INSERT INTO {} (external_id, embedding, metadata)
                VALUES ($1, $2, $3)
                ON CONFLICT (external_id) DO UPDATE SET
                    embedding = EXCLUDED.embedding,
                    metadata = EXCLUDED.metadata
                RETURNING id
                "#,
                self.table_name
            );
            
            let row = sqlx::query(&sql)
                .bind(external_id)
                .bind(vector)
                .bind(metadata)
                .fetch_one(&self.pool)
                .await
                .map_err(|e| EmbedVecError::PersistenceError(format!("Failed to insert vector: {}", e)))?;
            
            Ok(row.get("id"))
        }
        
        /// Search for similar vectors using pgvector's native search
        pub async fn search_vectors(
            &self,
            query: &[f32],
            k: usize,
            ef_search: Option<usize>,
        ) -> Result<Vec<(i64, String, f32, Option<serde_json::Value>)>> {
            let vector = Vector::from(query.to_vec());
            
            // Set ef_search for HNSW if provided
            if let Some(ef) = ef_search {
                let set_ef_sql = format!("SET hnsw.ef_search = {}", ef);
                sqlx::query(&set_ef_sql)
                    .execute(&self.pool)
                    .await
                    .map_err(|e| EmbedVecError::PersistenceError(format!("Failed to set ef_search: {}", e)))?;
            }
            
            let sql = format!(
                r#"
                SELECT id, external_id, 1 - (embedding <=> $1) as similarity, metadata
                FROM {}
                ORDER BY embedding <=> $1
                LIMIT $2
                "#,
                self.table_name
            );
            
            let rows = sqlx::query(&sql)
                .bind(vector)
                .bind(k as i64)
                .fetch_all(&self.pool)
                .await
                .map_err(|e| EmbedVecError::PersistenceError(format!("Failed to search vectors: {}", e)))?;
            
            let mut results = Vec::with_capacity(rows.len());
            for row in rows {
                let id: i64 = row.get("id");
                let external_id: String = row.get("external_id");
                let similarity: f32 = row.get("similarity");
                let metadata: Option<serde_json::Value> = row.get("metadata");
                results.push((id, external_id, similarity, metadata));
            }
            
            Ok(results)
        }
        
        /// Get vector count
        pub async fn count(&self) -> Result<usize> {
            let sql = format!("SELECT COUNT(*) as count FROM {}", self.table_name);
            let row = sqlx::query(&sql)
                .fetch_one(&self.pool)
                .await
                .map_err(|e| EmbedVecError::PersistenceError(format!("Failed to count vectors: {}", e)))?;
            
            let count: i64 = row.get("count");
            Ok(count as usize)
        }
        
        /// Delete a vector by external ID
        pub async fn delete_vector(&self, external_id: &str) -> Result<bool> {
            let sql = format!("DELETE FROM {} WHERE external_id = $1", self.table_name);
            let result = sqlx::query(&sql)
                .bind(external_id)
                .execute(&self.pool)
                .await
                .map_err(|e| EmbedVecError::PersistenceError(format!("Failed to delete vector: {}", e)))?;
            
            Ok(result.rows_affected() > 0)
        }
        
        /// Clear all vectors
        pub async fn clear(&self) -> Result<()> {
            let sql = format!("TRUNCATE TABLE {}", self.table_name);
            sqlx::query(&sql)
                .execute(&self.pool)
                .await
                .map_err(|e| EmbedVecError::PersistenceError(format!("Failed to clear table: {}", e)))?;
            
            Ok(())
        }
        
        /// Get the dimension
        pub fn dimension(&self) -> usize {
            self.dimension
        }
        
        /// Get the table name
        pub fn table_name(&self) -> &str {
            &self.table_name
        }
    }
    
    // Implement PersistenceBackend for key-value operations (metadata storage)
    impl PersistenceBackend for PgVectorBackend {
        fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
            // Use blocking runtime for sync interface
            let pool = self.pool.clone();
            let table_name = self.table_name.clone();
            let key = key.to_vec();
            
            let result = tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current().block_on(async {
                    let sql = format!("SELECT value FROM {}_metadata WHERE key = $1", table_name);
                    let row = sqlx::query(&sql)
                        .bind(&key)
                        .fetch_optional(&pool)
                        .await
                        .map_err(|e| EmbedVecError::PersistenceError(e.to_string()))?;
                    
                    Ok(row.map(|r| r.get::<Vec<u8>, _>("value")))
                })
            });
            
            result
        }
        
        fn set(&self, key: &[u8], value: &[u8]) -> Result<()> {
            let pool = self.pool.clone();
            let table_name = self.table_name.clone();
            let key = key.to_vec();
            let value = value.to_vec();
            
            tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current().block_on(async {
                    let sql = format!(
                        "INSERT INTO {}_metadata (key, value) VALUES ($1, $2) ON CONFLICT (key) DO UPDATE SET value = $2",
                        table_name
                    );
                    sqlx::query(&sql)
                        .bind(&key)
                        .bind(&value)
                        .execute(&pool)
                        .await
                        .map_err(|e| EmbedVecError::PersistenceError(e.to_string()))?;
                    
                    Ok(())
                })
            })
        }
        
        fn delete(&self, key: &[u8]) -> Result<()> {
            let pool = self.pool.clone();
            let table_name = self.table_name.clone();
            let key = key.to_vec();
            
            tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current().block_on(async {
                    let sql = format!("DELETE FROM {}_metadata WHERE key = $1", table_name);
                    sqlx::query(&sql)
                        .bind(&key)
                        .execute(&pool)
                        .await
                        .map_err(|e| EmbedVecError::PersistenceError(e.to_string()))?;
                    
                    Ok(())
                })
            })
        }
        
        fn contains(&self, key: &[u8]) -> Result<bool> {
            self.get(key).map(|opt| opt.is_some())
        }
        
        fn scan_prefix(&self, prefix: &[u8]) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
            let pool = self.pool.clone();
            let table_name = self.table_name.clone();
            let prefix = prefix.to_vec();
            
            tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current().block_on(async {
                    // PostgreSQL doesn't have native prefix scan for bytea, so we use LIKE with escape
                    let sql = format!(
                        "SELECT key, value FROM {}_metadata WHERE key >= $1 AND key < $2",
                        table_name
                    );
                    
                    // Calculate the upper bound for prefix scan
                    let mut upper_bound = prefix.clone();
                    if let Some(last) = upper_bound.last_mut() {
                        *last = last.saturating_add(1);
                    }
                    
                    let rows = sqlx::query(&sql)
                        .bind(&prefix)
                        .bind(&upper_bound)
                        .fetch_all(&pool)
                        .await
                        .map_err(|e| EmbedVecError::PersistenceError(e.to_string()))?;
                    
                    let results: Vec<(Vec<u8>, Vec<u8>)> = rows
                        .iter()
                        .map(|r| (r.get::<Vec<u8>, _>("key"), r.get::<Vec<u8>, _>("value")))
                        .collect();
                    
                    Ok(results)
                })
            })
        }
        
        fn flush(&self) -> Result<()> {
            // PostgreSQL handles durability automatically
            Ok(())
        }
        
        fn backend_type(&self) -> BackendType {
            BackendType::PgVector
        }
    }
}

#[cfg(feature = "persistence-pgvector")]
pub use pgvector_backend::PgVectorBackend;

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
        BackendType::PgVector => {
            Err(EmbedVecError::PersistenceError(
                "PgVector backend requires async initialization. Use create_backend_async instead.".to_string()
            ))
        }
    }
}

/// Create a persistence backend asynchronously (required for pgvector)
#[cfg(all(feature = "async", feature = "persistence-pgvector"))]
pub async fn create_backend_async(config: &BackendConfig) -> Result<Box<dyn PersistenceBackend>> {
    match config.backend_type {
        BackendType::PgVector => {
            let backend = PgVectorBackend::connect(config).await?;
            Ok(Box::new(backend))
        }
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
