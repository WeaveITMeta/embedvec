//! # embedvec — High-Performance Embedded Vector Database
//!
//! [![crates.io](https://img.shields.io/crates/v/embedvec.svg)](https://crates.io/crates/embedvec)
//! [![docs.rs](https://docs.rs/embedvec/badge.svg)](https://docs.rs/embedvec)
//! [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
//!
//! **Fast, lightweight, in-process vector database** with HNSW indexing, SIMD-accelerated
//! distance calculations, metadata filtering, E8 lattice quantization, and optional
//! persistence via Sled or RocksDB.
//!
//! ## Why embedvec?
//!
//! - **Pure Rust** — No C++ dependencies (unless using RocksDB), safe and portable
//! - **Blazing Fast** — AVX2/FMA SIMD acceleration, optimized HNSW with O(1) lookups
//! - **Memory Efficient** — E8 quantization provides 4-6× compression with <5% recall loss
//! - **Flexible Persistence** — Choose between Sled (pure Rust) or RocksDB (high performance)
//! - **Production Ready** — Async API, metadata filtering, batch operations
//!
//! ## Benchmarks (768-dim vectors, 10k dataset)
//!
//! | Operation | Time | Throughput |
//! |-----------|------|------------|
//! | **Search (ef=32)** | 3.0 ms | 3,300 queries/sec |
//! | **Search (ef=64)** | 4.9 ms | 2,000 queries/sec |
//! | **Search (ef=128)** | 16.1 ms | 620 queries/sec |
//! | **Insert (768-dim)** | 25.5 ms/100 | 3,900 vectors/sec |
//! | **Distance (cosine)** | 122 ns/pair | 8.2M ops/sec |
//! | **Distance (dot)** | 91 ns/pair | 11M ops/sec |
//!
//! *Benchmarks on AMD Ryzen 9 / Intel i9, AVX2 enabled. Run `cargo bench` to reproduce.*
//!
//! ## Features
//!
//! | Feature | Description |
//! |---------|-------------|
//! | **HNSW Indexing** | Hierarchical Navigable Small World graph for O(log n) ANN search |
//! | **SIMD Distance** | AVX2/FMA accelerated cosine, euclidean, dot product |
//! | **E8 Quantization** | Lattice-based compression (4-6× memory reduction) |
//! | **Metadata Filtering** | Composable filters: eq, gt, lt, contains, AND/OR/NOT |
//! | **Dual Persistence** | Sled (pure Rust) or RocksDB (high performance) |
//! | **Async API** | Tokio-compatible async operations |
//! | **Python Bindings** | PyO3-based interop (feature-gated) |
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use embedvec::{Distance, EmbedVec, FilterExpr};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Create in-memory database
//!     let mut db = EmbedVec::new(768, Distance::Cosine, 32, 200).await?;
//!     
//!     // Add vectors with metadata
//!     db.add(&vec![0.1; 768], serde_json::json!({"doc_id": "123", "category": "tech"})).await?;
//!     
//!     // Search with optional filter
//!     let filter = FilterExpr::eq("category", "tech");
//!     let results = db.search(&vec![0.15; 768], 10, 100, Some(filter)).await?;
//!     
//!     for hit in results {
//!         println!("id: {}, score: {:.4}", hit.id, hit.score);
//!     }
//!     Ok(())
//! }
//! ```
//!
//! ## With Persistence
//!
//! ```rust,no_run
//! use embedvec::{Distance, EmbedVec, BackendConfig, BackendType};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Sled backend (default, pure Rust)
//!     let db = EmbedVec::with_persistence("/tmp/vectors.db", 768, Distance::Cosine, 32, 200).await?;
//!     
//!     // Or RocksDB for higher performance (requires persistence-rocksdb feature)
//!     // let config = BackendConfig::new("/tmp/vectors.db").backend(BackendType::RocksDb);
//!     // let db = EmbedVec::with_backend(config, 768, Distance::Cosine, 32, 200).await?;
//!     Ok(())
//! }
//! ```
//!
//! ## E8 Quantization (4-6× Memory Savings)
//!
//! ```rust,no_run
//! use embedvec::{EmbedVec, Distance, Quantization};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let db = EmbedVec::builder()
//!         .dimension(768)
//!         .metric(Distance::Cosine)
//!         .quantization(Quantization::e8_default())  // ~1.25 bits/dim
//!         .build()
//!         .await?;
//!     Ok(())
//! }
//! ```
//!
//! ## Feature Flags
//!
//! | Flag | Description | Default |
//! |------|-------------|---------|
//! | `persistence-sled` | Sled persistence backend | ✓ |
//! | `persistence-rocksdb` | RocksDB persistence backend | ✗ |
//! | `async` | Tokio async API | ✓ |
//! | `python` | PyO3 Python bindings | ✗ |
//! | `simd` | Explicit SIMD (auto-detected) | ✗ |
//!
//! ## Memory Usage (768-dim vectors)
//!
//! | Mode | Bits/Dim | Memory/Vector | 1M Vectors |
//! |------|----------|---------------|------------|
//! | Raw (f32) | 32 | 3.1 KB | ~3.1 GB |
//! | E8 10-bit | ~1.25 | ~220 B | ~220 MB |
//!
//! ## License
//!
//! MIT OR Apache-2.0

#![warn(missing_docs)]
#![warn(rustdoc::missing_crate_level_docs)]

// WASM global allocator
#[cfg(feature = "wasm")]
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

pub mod distance;
pub mod e8;
pub mod error;
pub mod filter;
pub mod hnsw;
pub mod metadata;
#[cfg(any(feature = "persistence-sled", feature = "persistence-rocksdb"))]
pub mod persistence;
pub mod quantization;
pub mod storage;

#[cfg(feature = "python")]
pub mod python;

// Re-exports for convenient API access
pub use distance::Distance;
pub use e8::{E8Codec, HadamardTransform};
pub use error::{EmbedVecError, Result};
pub use filter::FilterExpr;
pub use hnsw::HnswIndex;
pub use metadata::Metadata;
#[cfg(any(feature = "persistence-sled", feature = "persistence-rocksdb"))]
pub use persistence::{BackendConfig, BackendType, PersistenceBackend};
pub use quantization::Quantization;
pub use storage::VectorStorage;

use ordered_float::OrderedFloat;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Search result hit containing vector ID, similarity score, and payload
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hit {
    /// Unique identifier of the vector
    pub id: usize,
    /// Similarity/distance score (interpretation depends on metric)
    pub score: f32,
    /// Associated metadata payload
    pub payload: Metadata,
}

impl Hit {
    /// Create a new Hit
    pub fn new(id: usize, score: f32, payload: Metadata) -> Self {
        Self { id, score, payload }
    }
}

/// Builder for configuring EmbedVec instances
#[derive(Debug, Clone)]
pub struct EmbedVecBuilder {
    dimension: usize,
    distance: Distance,
    m: usize,
    ef_construction: usize,
    quantization: Quantization,
    #[cfg(any(feature = "persistence-sled", feature = "persistence-rocksdb"))]
    persistence_config: Option<persistence::BackendConfig>,
}

impl EmbedVecBuilder {
    /// Create a new builder with required dimension
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            distance: Distance::Cosine,
            m: 16,
            ef_construction: 200,
            quantization: Quantization::None,
            #[cfg(any(feature = "persistence-sled", feature = "persistence-rocksdb"))]
            persistence_config: None,
        }
    }

    /// Set the dimension of vectors
    pub fn dimension(mut self, dim: usize) -> Self {
        self.dimension = dim;
        self
    }

    /// Set the distance metric
    pub fn metric(mut self, distance: Distance) -> Self {
        self.distance = distance;
        self
    }

    /// Set HNSW M parameter (connections per layer)
    pub fn m(mut self, m: usize) -> Self {
        self.m = m;
        self
    }

    /// Set HNSW ef_construction parameter
    pub fn ef_construction(mut self, ef: usize) -> Self {
        self.ef_construction = ef;
        self
    }

    /// Set quantization mode
    pub fn quantization(mut self, quant: Quantization) -> Self {
        self.quantization = quant;
        self
    }

    /// Set persistence path for on-disk storage (uses Sled by default)
    #[cfg(any(feature = "persistence-sled", feature = "persistence-rocksdb"))]
    pub fn persistence(mut self, path: impl Into<String>) -> Self {
        self.persistence_config = Some(persistence::BackendConfig::new(path));
        self
    }
    
    /// Set persistence with full backend configuration
    #[cfg(any(feature = "persistence-sled", feature = "persistence-rocksdb"))]
    pub fn persistence_config(mut self, config: persistence::BackendConfig) -> Self {
        self.persistence_config = Some(config);
        self
    }

    /// Build the EmbedVec instance
    #[cfg(feature = "async")]
    pub async fn build(self) -> Result<EmbedVec> {
        EmbedVec::from_builder(self).await
    }

    /// Build the EmbedVec instance (sync version)
    #[cfg(not(feature = "async"))]
    pub fn build(self) -> Result<EmbedVec> {
        EmbedVec::new_internal(
            self.dimension,
            self.distance,
            self.m,
            self.ef_construction,
            self.quantization,
            #[cfg(any(feature = "persistence-sled", feature = "persistence-rocksdb"))]
            self.persistence_config,
        )
    }
}

/// Main embedded vector database struct
///
/// Provides HNSW-based approximate nearest neighbor search with optional
/// E8 lattice quantization for memory efficiency.
pub struct EmbedVec {
    /// Vector dimension
    dimension: usize,
    /// Distance metric
    distance: Distance,
    /// HNSW index
    pub index: Arc<RwLock<HnswIndex>>,
    /// Vector storage (raw or quantized)
    pub storage: Arc<RwLock<VectorStorage>>,
    /// Metadata storage
    pub metadata: Arc<RwLock<Vec<Metadata>>>,
    /// Quantization configuration
    quantization: Quantization,
    /// E8 codec (if quantization enabled)
    e8_codec: Option<E8Codec>,
    /// Persistence backend (sled or rocksdb)
    #[cfg(any(feature = "persistence-sled", feature = "persistence-rocksdb"))]
    backend: Option<Box<dyn persistence::PersistenceBackend>>,
}

impl EmbedVec {
    /// Create a new in-memory EmbedVec instance
    ///
    /// # Arguments
    /// * `dim` - Vector dimension (e.g., 768 for many LLM embeddings)
    /// * `distance` - Distance metric (Cosine, Euclidean, DotProduct)
    /// * `m` - HNSW M parameter (connections per layer, typically 16-64)
    /// * `ef_construction` - HNSW construction parameter (typically 100-500)
    ///
    /// # Example
    /// ```rust,no_run
    /// use embedvec::{EmbedVec, Distance};
    ///
    /// #[tokio::main]
    /// async fn main() {
    ///     let db = EmbedVec::new(768, Distance::Cosine, 32, 200).await.unwrap();
    /// }
    /// ```
    #[cfg(all(feature = "async", any(feature = "persistence-sled", feature = "persistence-rocksdb")))]
    pub async fn new(
        dim: usize,
        distance: Distance,
        m: usize,
        ef_construction: usize,
    ) -> Result<Self> {
        Self::new_internal(dim, distance, m, ef_construction, Quantization::None, None)
    }
    
    #[cfg(all(feature = "async", not(any(feature = "persistence-sled", feature = "persistence-rocksdb"))))]
    pub async fn new(
        dim: usize,
        distance: Distance,
        m: usize,
        ef_construction: usize,
    ) -> Result<Self> {
        Self::new_internal(dim, distance, m, ef_construction, Quantization::None)
    }

    /// Create a new EmbedVec with persistence (uses Sled by default)
    #[cfg(all(feature = "async", any(feature = "persistence-sled", feature = "persistence-rocksdb")))]
    pub async fn with_persistence(
        path: impl AsRef<std::path::Path>,
        dim: usize,
        distance: Distance,
        m: usize,
        ef_construction: usize,
    ) -> Result<Self> {
        let path_str = path.as_ref().to_string_lossy().to_string();
        let config = persistence::BackendConfig::new(path_str);
        Self::new_internal(
            dim,
            distance,
            m,
            ef_construction,
            Quantization::None,
            Some(config),
        )
    }
    
    /// Create a new EmbedVec with a specific persistence backend
    #[cfg(all(feature = "async", any(feature = "persistence-sled", feature = "persistence-rocksdb")))]
    pub async fn with_backend(
        config: persistence::BackendConfig,
        dim: usize,
        distance: Distance,
        m: usize,
        ef_construction: usize,
    ) -> Result<Self> {
        Self::new_internal(
            dim,
            distance,
            m,
            ef_construction,
            Quantization::None,
            Some(config),
        )
    }

    /// Create EmbedVec from builder configuration
    #[cfg(feature = "async")]
    async fn from_builder(builder: EmbedVecBuilder) -> Result<Self> {
        Self::new_internal(
            builder.dimension,
            builder.distance,
            builder.m,
            builder.ef_construction,
            builder.quantization,
            #[cfg(any(feature = "persistence-sled", feature = "persistence-rocksdb"))]
            builder.persistence_config,
        )
    }

    /// Internal constructor (public for Python bindings)
    #[cfg(any(feature = "persistence-sled", feature = "persistence-rocksdb"))]
    pub fn new_internal(
        dim: usize,
        distance: Distance,
        m: usize,
        ef_construction: usize,
        quantization: Quantization,
        persistence_config: Option<persistence::BackendConfig>,
    ) -> Result<Self> {
        if dim == 0 {
            return Err(EmbedVecError::InvalidDimension(dim));
        }

        let index = HnswIndex::new(m, ef_construction, distance);
        let storage = VectorStorage::new(dim, quantization.clone());

        let e8_codec = match &quantization {
            Quantization::None => None,
            Quantization::E8 {
                bits_per_block,
                use_hadamard,
                random_seed,
            } => Some(E8Codec::new(dim, *bits_per_block, *use_hadamard, *random_seed)),
        };

        let backend = if let Some(config) = persistence_config {
            Some(persistence::create_backend(&config)?)
        } else {
            None
        };

        Ok(Self {
            dimension: dim,
            distance,
            index: Arc::new(RwLock::new(index)),
            storage: Arc::new(RwLock::new(storage)),
            metadata: Arc::new(RwLock::new(Vec::new())),
            quantization,
            e8_codec,
            backend,
        })
    }
    
    /// Internal constructor without persistence
    #[cfg(not(any(feature = "persistence-sled", feature = "persistence-rocksdb")))]
    pub fn new_internal(
        dim: usize,
        distance: Distance,
        m: usize,
        ef_construction: usize,
        quantization: Quantization,
    ) -> Result<Self> {
        if dim == 0 {
            return Err(EmbedVecError::InvalidDimension(dim));
        }

        let index = HnswIndex::new(m, ef_construction, distance);
        let storage = VectorStorage::new(dim, quantization.clone());

        let e8_codec = match &quantization {
            Quantization::None => None,
            Quantization::E8 {
                bits_per_block,
                use_hadamard,
                random_seed,
            } => Some(E8Codec::new(dim, *bits_per_block, *use_hadamard, *random_seed)),
        };

        Ok(Self {
            dimension: dim,
            distance,
            index: Arc::new(RwLock::new(index)),
            storage: Arc::new(RwLock::new(storage)),
            metadata: Arc::new(RwLock::new(Vec::new())),
            quantization,
            e8_codec,
        })
    }

    /// Get a builder for configuring EmbedVec
    pub fn builder() -> EmbedVecBuilder {
        EmbedVecBuilder::new(768) // Default dimension
    }

    /// Add a single vector with metadata
    ///
    /// # Arguments
    /// * `vector` - The embedding vector (must match configured dimension)
    /// * `payload` - Associated metadata (JSON-compatible)
    ///
    /// # Returns
    /// The assigned vector ID
    #[cfg(feature = "async")]
    pub async fn add(&mut self, vector: &[f32], payload: impl Into<Metadata>) -> Result<usize> {
        self.add_internal(vector, payload.into())
    }

    /// Add multiple vectors with metadata in batch
    ///
    /// # Arguments
    /// * `vectors` - Slice of embedding vectors
    /// * `payloads` - Associated metadata for each vector
    #[cfg(feature = "async")]
    pub async fn add_many(
        &mut self,
        vectors: &[Vec<f32>],
        payloads: Vec<impl Into<Metadata>>,
    ) -> Result<()> {
        if vectors.len() != payloads.len() {
            return Err(EmbedVecError::MismatchedLengths {
                vectors: vectors.len(),
                payloads: payloads.len(),
            });
        }

        for (vector, payload) in vectors.iter().zip(payloads.into_iter()) {
            self.add_internal(vector, payload.into())?;
        }

        Ok(())
    }

    /// Internal add implementation (public for Python bindings)
    pub fn add_internal(&mut self, vector: &[f32], payload: Metadata) -> Result<usize> {
        if vector.len() != self.dimension {
            return Err(EmbedVecError::DimensionMismatch {
                expected: self.dimension,
                got: vector.len(),
            });
        }

        // Normalize if using cosine distance
        let processed_vector = if self.distance == Distance::Cosine {
            normalize_vector(vector)
        } else {
            vector.to_vec()
        };

        // Store vector (quantized or raw)
        let id = {
            let mut storage = self.storage.write();
            storage.add(&processed_vector, self.e8_codec.as_ref())?
        };

        // Store metadata
        {
            let mut meta = self.metadata.write();
            if id >= meta.len() {
                meta.resize(id + 1, Metadata::default());
            }
            meta[id] = payload;
        }

        // Add to HNSW index
        {
            let mut index = self.index.write();
            let storage = self.storage.read();
            index.insert(id, &processed_vector, &storage, self.e8_codec.as_ref())?;
        }

        Ok(id)
    }

    /// Search for nearest neighbors
    ///
    /// # Arguments
    /// * `query` - Query vector
    /// * `k` - Number of results to return
    /// * `ef_search` - Search parameter (higher = better recall, slower)
    /// * `filter` - Optional metadata filter expression
    ///
    /// # Returns
    /// Vector of Hit results sorted by similarity
    #[cfg(feature = "async")]
    pub async fn search(
        &self,
        query: &[f32],
        k: usize,
        ef_search: usize,
        filter: Option<FilterExpr>,
    ) -> Result<Vec<Hit>> {
        self.search_internal(query, k, ef_search, filter)
    }

    /// Internal search implementation (public for Python bindings)
    pub fn search_internal(
        &self,
        query: &[f32],
        k: usize,
        ef_search: usize,
        filter: Option<FilterExpr>,
    ) -> Result<Vec<Hit>> {
        if query.len() != self.dimension {
            return Err(EmbedVecError::DimensionMismatch {
                expected: self.dimension,
                got: query.len(),
            });
        }

        // Normalize query if using cosine distance
        let processed_query = if self.distance == Distance::Cosine {
            normalize_vector(query)
        } else {
            query.to_vec()
        };

        // Search HNSW index
        let candidates = {
            let index = self.index.read();
            let storage = self.storage.read();
            index.search(
                &processed_query,
                k,
                ef_search,
                &storage,
                self.e8_codec.as_ref(),
            )?
        };

        // Apply filter and collect results
        let metadata = self.metadata.read();
        let mut results: Vec<Hit> = candidates
            .into_iter()
            .filter_map(|(id, score)| {
                let payload = metadata.get(id)?.clone();

                // Apply filter if present
                if let Some(ref f) = filter {
                    if !f.matches(&payload) {
                        return None;
                    }
                }

                Some(Hit::new(id, score, payload))
            })
            .take(k)
            .collect();

        // Sort by score (lower is better for distance metrics)
        results.sort_by_key(|h| OrderedFloat(h.score));

        Ok(results)
    }

    /// Get the number of vectors in the database
    #[cfg(feature = "async")]
    pub async fn len(&self) -> usize {
        self.storage.read().len()
    }

    /// Check if the database is empty
    #[cfg(feature = "async")]
    pub async fn is_empty(&self) -> bool {
        self.storage.read().is_empty()
    }

    /// Clear all vectors and metadata
    #[cfg(feature = "async")]
    pub async fn clear(&mut self) -> Result<()> {
        {
            let mut storage = self.storage.write();
            storage.clear();
        }
        {
            let mut metadata = self.metadata.write();
            metadata.clear();
        }
        {
            let mut index = self.index.write();
            index.clear();
        }
        Ok(())
    }

    /// Flush changes to disk (if persistence enabled)
    #[cfg(all(feature = "async", feature = "persistence"))]
    pub async fn flush(&mut self) -> Result<()> {
        if let Some(ref db) = self.db {
            db.flush()
                .map_err(|e| EmbedVecError::PersistenceError(e.to_string()))?;
        }
        Ok(())
    }

    /// Get current quantization mode
    pub fn quantization(&self) -> &Quantization {
        &self.quantization
    }

    /// Set quantization mode (requires re-indexing)
    #[cfg(feature = "async")]
    pub async fn set_quantization(&mut self, quant: Quantization) -> Result<()> {
        self.quantization = quant.clone();
        self.e8_codec = match &quant {
            Quantization::None => None,
            Quantization::E8 {
                bits_per_block,
                use_hadamard,
                random_seed,
            } => Some(E8Codec::new(
                self.dimension,
                *bits_per_block,
                *use_hadamard,
                *random_seed,
            )),
        };

        // Re-quantize existing vectors
        let mut storage = self.storage.write();
        storage.set_quantization(quant, self.e8_codec.as_ref())?;

        Ok(())
    }

    /// Get vector dimension
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Get distance metric
    pub fn distance(&self) -> Distance {
        self.distance
    }
}

/// Normalize a vector to unit length
fn normalize_vector(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-10 {
        v.iter().map(|x| x / norm).collect()
    } else {
        v.to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_basic_operations() {
        let mut db = EmbedVec::new(4, Distance::Cosine, 16, 100).await.unwrap();

        let id = db
            .add(&[1.0, 0.0, 0.0, 0.0], serde_json::json!({"test": "value"}))
            .await
            .unwrap();
        assert_eq!(id, 0);

        let results = db.search(&[1.0, 0.0, 0.0, 0.0], 1, 50, None).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, 0);
    }

    #[tokio::test]
    async fn test_dimension_mismatch() {
        let mut db = EmbedVec::new(4, Distance::Cosine, 16, 100).await.unwrap();

        let result = db
            .add(&[1.0, 0.0, 0.0], serde_json::json!({}))
            .await;
        assert!(result.is_err());
    }
}
