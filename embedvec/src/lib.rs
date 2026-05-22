//! # embedvec — High-Performance Embedded Vector Database
//!
//! [![crates.io](https://img.shields.io/crates/v/embedvec.svg)](https://crates.io/crates/embedvec)
//! [![docs.rs](https://docs.rs/embedvec/badge.svg)](https://docs.rs/embedvec)
//! [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
//!
//! **Fast, lightweight, in-process vector database** with HNSW indexing, SIMD-accelerated
//! distance calculations, metadata filtering, E8 and H4 lattice quantization, and optional
//! persistence via Fjall (default), Sled, or RocksDB.
//!
//! ## Why embedvec?
//!
//! - **Pure Rust** — No C++ dependencies (unless using RocksDB), safe and portable
//! - **Blazing Fast** — AVX2/FMA SIMD acceleration, optimized HNSW with O(1) lookups
//! - **Memory Efficient** — E8/H4 quantization provides 4-16× compression with <5% recall loss
//! - **Flexible Persistence** — Fjall (default, pure Rust LSM), Sled (pure Rust), or RocksDB (high performance)
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
//! | **E8 Quantization** | 8D lattice compression (4-6× memory reduction) |
//! | **H4 Quantization** | 4D 600-cell compression (~15× memory reduction) |
//! | **Metadata Filtering** | Composable filters: eq, gt, lt, contains, AND/OR/NOT |
//! | **Flexible Persistence** | Fjall (default, pure Rust LSM), Sled (pure Rust), or RocksDB (high performance) |
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
//!     // Fjall backend (default, pure Rust LSM-tree)
//!     let db = EmbedVec::with_persistence("/tmp/vectors.db", 768, Distance::Cosine, 32, 200).await?;
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
//! ## H4 Quantization (~15× Memory Savings)
//!
//! ```rust,no_run
//! use embedvec::{EmbedVec, Distance, Quantization};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let db = EmbedVec::builder()
//!         .dimension(768)
//!         .metric(Distance::Cosine)
//!         .quantization(Quantization::h4_default())  // ~1.73 bits/dim, 4D 600-cell
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
//! | `persistence-fjall` | Fjall persistence backend (pure Rust LSM) | ✓ |
//! | `persistence-sled` | Sled persistence backend | ✗ |
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
//! | H4 | ~1.73 | ~196 B | ~196 MB |
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
pub mod h4;
pub mod hnsw;
pub mod metadata;
#[cfg(any(feature = "persistence-fjall", feature = "persistence-sled", feature = "persistence-rocksdb"))]
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
pub use h4::{H4Codec, hadamard4_inplace};
pub use hnsw::HnswIndex;
pub use metadata::Metadata;
#[cfg(any(feature = "persistence-fjall", feature = "persistence-sled", feature = "persistence-rocksdb"))]
pub use persistence::{BackendConfig, BackendType, PersistenceBackend};
pub use quantization::Quantization;
pub use storage::VectorStorage;

use ordered_float::OrderedFloat;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
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
    #[cfg(any(feature = "persistence-fjall", feature = "persistence-sled", feature = "persistence-rocksdb"))]
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
            #[cfg(any(feature = "persistence-fjall", feature = "persistence-sled", feature = "persistence-rocksdb"))]
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

    /// Set persistence path for on-disk storage (uses Fjall by default)
    #[cfg(any(feature = "persistence-fjall", feature = "persistence-sled", feature = "persistence-rocksdb"))]
    pub fn persistence(mut self, path: impl Into<String>) -> Self {
        self.persistence_config = Some(persistence::BackendConfig::new(path));
        self
    }

    /// Set persistence with full backend configuration
    #[cfg(any(feature = "persistence-fjall", feature = "persistence-sled", feature = "persistence-rocksdb"))]
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
            #[cfg(any(feature = "persistence-fjall", feature = "persistence-sled", feature = "persistence-rocksdb"))]
            self.persistence_config,
        )
    }
}

/// Main embedded vector database struct
///
/// Provides HNSW-based approximate nearest neighbor search with optional
/// E8 or H4 lattice quantization for memory efficiency.
pub struct EmbedVec {
    /// Vector dimension
    dimension: usize,
    /// Distance metric
    distance: Distance,
    /// HNSW index
    pub index: Arc<RwLock<HnswIndex>>,
    /// Vector storage (raw, E8-quantized, or H4-quantized)
    pub storage: Arc<RwLock<VectorStorage>>,
    /// Metadata storage
    pub metadata: Arc<RwLock<Vec<Metadata>>>,
    /// Quantization configuration
    quantization: Quantization,
    /// E8 codec (present when quantization = E8)
    e8_codec: Option<E8Codec>,
    /// H4 codec (present when quantization = H4)
    h4_codec: Option<H4Codec>,
    /// Soft-deleted vector ids — excluded from search results and `len()`.
    /// Their slots stay in storage (as graph waypoints) until the next reopen.
    deleted: Arc<RwLock<HashSet<usize>>>,
    /// Persistence backend (fjall, sled, or rocksdb)
    #[cfg(any(feature = "persistence-fjall", feature = "persistence-sled", feature = "persistence-rocksdb"))]
    backend: Option<Box<dyn persistence::PersistenceBackend>>,
}

// =============================================================================
// On-disk persistence format
// =============================================================================

/// Backend key holding the self-describing database header.
#[cfg(any(feature = "persistence-fjall", feature = "persistence-sled", feature = "persistence-rocksdb"))]
const PERSIST_HEADER_KEY: &[u8] = b"__embedvec_header__";

/// Backend key prefix for per-vector records. Ids are zero-padded so a
/// lexicographic prefix scan returns records in ascending id order.
#[cfg(any(feature = "persistence-fjall", feature = "persistence-sled", feature = "persistence-rocksdb"))]
const PERSIST_REC_PREFIX: &[u8] = b"rec:";

/// On-disk format version (bump on incompatible layout changes).
#[cfg(any(feature = "persistence-fjall", feature = "persistence-sled", feature = "persistence-rocksdb"))]
const PERSIST_VERSION: u32 = 1;

/// Self-describing header so a persisted database reopens with the exact
/// configuration it was created with, regardless of constructor arguments.
#[cfg(any(feature = "persistence-fjall", feature = "persistence-sled", feature = "persistence-rocksdb"))]
#[derive(Serialize, Deserialize)]
struct PersistHeader {
    version: u32,
    dimension: usize,
    distance: Distance,
    quantization: Quantization,
    m: usize,
    ef_construction: usize,
    /// Highest id slot ever allocated (incl. deleted) at last flush — lets a
    /// reopen recreate trailing tombstones so deleted ids are never reused.
    #[serde(default)]
    high_water_mark: usize,
}

/// One persisted vector: its (compact) stored form plus its metadata.
#[cfg(any(feature = "persistence-fjall", feature = "persistence-sled", feature = "persistence-rocksdb"))]
#[derive(Serialize, Deserialize)]
struct PersistedRecord {
    stored: storage::StoredVector,
    meta: Metadata,
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
    #[cfg(all(feature = "async", any(feature = "persistence-fjall", feature = "persistence-sled", feature = "persistence-rocksdb")))]
    pub async fn new(
        dim: usize,
        distance: Distance,
        m: usize,
        ef_construction: usize,
    ) -> Result<Self> {
        Self::new_internal(dim, distance, m, ef_construction, Quantization::None, None)
    }

    #[cfg(all(feature = "async", not(any(feature = "persistence-fjall", feature = "persistence-sled", feature = "persistence-rocksdb"))))]
    pub async fn new(
        dim: usize,
        distance: Distance,
        m: usize,
        ef_construction: usize,
    ) -> Result<Self> {
        Self::new_internal(dim, distance, m, ef_construction, Quantization::None)
    }

    /// Create a new EmbedVec with persistence (uses Fjall by default)
    #[cfg(all(feature = "async", any(feature = "persistence-fjall", feature = "persistence-sled", feature = "persistence-rocksdb")))]
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
    #[cfg(all(feature = "async", any(feature = "persistence-fjall", feature = "persistence-sled", feature = "persistence-rocksdb")))]
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
            #[cfg(any(feature = "persistence-fjall", feature = "persistence-sled", feature = "persistence-rocksdb"))]
            builder.persistence_config,
        )
    }

    /// Internal constructor (public for Python bindings) — with persistence
    #[cfg(any(feature = "persistence-fjall", feature = "persistence-sled", feature = "persistence-rocksdb"))]
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

        let mut index = HnswIndex::new(m, ef_construction, distance);
        let storage = VectorStorage::new(dim, quantization.clone());

        let e8_codec = match &quantization {
            Quantization::E8 { bits_per_block, use_hadamard, random_seed } =>
                Some(E8Codec::new(dim, *bits_per_block, *use_hadamard, *random_seed)),
            _ => None,
        };

        let h4_codec = match &quantization {
            Quantization::H4 { use_hadamard, random_seed } =>
                Some(H4Codec::new(dim, *use_hadamard, *random_seed)),
            _ => None,
        };

        // The index needs the H4 codec to decode H4 vectors during distance
        // computation (E8 is threaded per-call; raw uses the zero-copy path).
        index.set_h4_codec(h4_codec.clone());

        let backend = if let Some(config) = persistence_config {
            Some(persistence::create_backend(&config)?)
        } else {
            None
        };

        let mut db = Self {
            dimension: dim,
            distance,
            index: Arc::new(RwLock::new(index)),
            storage: Arc::new(RwLock::new(storage)),
            metadata: Arc::new(RwLock::new(Vec::new())),
            quantization,
            e8_codec,
            h4_codec,
            deleted: Arc::new(RwLock::new(HashSet::new())),
            backend,
        };

        // Reload any previously persisted vectors (and adopt their config), or
        // record the header for a fresh store so it reopens consistently.
        db.load_from_backend()?;

        Ok(db)
    }

    /// Internal constructor without persistence
    #[cfg(not(any(feature = "persistence-fjall", feature = "persistence-sled", feature = "persistence-rocksdb")))]
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

        let mut index = HnswIndex::new(m, ef_construction, distance);
        let storage = VectorStorage::new(dim, quantization.clone());

        let e8_codec = match &quantization {
            Quantization::E8 { bits_per_block, use_hadamard, random_seed } =>
                Some(E8Codec::new(dim, *bits_per_block, *use_hadamard, *random_seed)),
            _ => None,
        };

        let h4_codec = match &quantization {
            Quantization::H4 { use_hadamard, random_seed } =>
                Some(H4Codec::new(dim, *use_hadamard, *random_seed)),
            _ => None,
        };

        // The index needs the H4 codec to decode H4 vectors during distance
        // computation (E8 is threaded per-call; raw uses the zero-copy path).
        index.set_h4_codec(h4_codec.clone());

        Ok(Self {
            dimension: dim,
            distance,
            index: Arc::new(RwLock::new(index)),
            storage: Arc::new(RwLock::new(storage)),
            metadata: Arc::new(RwLock::new(Vec::new())),
            quantization,
            e8_codec,
            h4_codec,
            deleted: Arc::new(RwLock::new(HashSet::new())),
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

        #[cfg(any(feature = "persistence-fjall", feature = "persistence-sled", feature = "persistence-rocksdb"))]
        let mut new_ids: Vec<usize> = Vec::with_capacity(vectors.len());

        for (vector, payload) in vectors.iter().zip(payloads.into_iter()) {
            let _id = self.insert_in_memory(vector, payload.into())?;
            #[cfg(any(feature = "persistence-fjall", feature = "persistence-sled", feature = "persistence-rocksdb"))]
            new_ids.push(_id);
        }

        // Persist the whole batch in a single atomic write (fast on Fjall/Sled).
        #[cfg(any(feature = "persistence-fjall", feature = "persistence-sled", feature = "persistence-rocksdb"))]
        self.persist_records(&new_ids)?;

        Ok(())
    }

    /// Internal add implementation (public for Python bindings)
    pub fn add_internal(&mut self, vector: &[f32], payload: Metadata) -> Result<usize> {
        let id = self.insert_in_memory(vector, payload)?;

        // Persist the single record (no-op when no backend is configured).
        #[cfg(any(feature = "persistence-fjall", feature = "persistence-sled", feature = "persistence-rocksdb"))]
        self.persist_record(id)?;

        Ok(id)
    }

    /// Insert a vector + metadata into the in-memory storage, metadata table,
    /// and HNSW index. Does not touch the persistence backend.
    fn insert_in_memory(&mut self, vector: &[f32], payload: Metadata) -> Result<usize> {
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
            storage.add(&processed_vector, self.e8_codec.as_ref(), self.h4_codec.as_ref())?
        };

        // Store metadata
        {
            let mut meta = self.metadata.write();
            if id >= meta.len() {
                meta.resize(id + 1, Metadata::default());
            }
            meta[id] = payload;
        }

        // Add to HNSW index (HNSW currently uses E8 codec path; H4 vectors inserted raw)
        {
            let mut index = self.index.write();
            let storage = self.storage.read();
            index.insert(id, &processed_vector, &storage, self.e8_codec.as_ref())?;
        }

        Ok(id)
    }

    /// Delete a vector by its id.
    ///
    /// Soft delete: the id is excluded from future searches and from `len()`,
    /// its metadata is cleared, and its record is removed from the persistence
    /// backend. The vector stays in the in-memory graph as a routing waypoint
    /// until the next reopen, when the index is rebuilt without it. Ids of
    /// surviving vectors remain stable and deleted ids are not reused.
    ///
    /// Returns `true` if the id existed and was newly deleted, or `false` if it
    /// was out of range or already deleted.
    #[cfg(feature = "async")]
    pub async fn delete(&mut self, id: usize) -> Result<bool> {
        self.delete_internal(id)
    }

    /// Delete many vectors by id. Returns the number actually removed.
    #[cfg(feature = "async")]
    pub async fn delete_many(&mut self, ids: &[usize]) -> Result<usize> {
        let mut removed = 0;
        for &id in ids {
            if self.delete_internal(id)? {
                removed += 1;
            }
        }
        Ok(removed)
    }

    /// Internal delete implementation (public for Python bindings).
    pub fn delete_internal(&mut self, id: usize) -> Result<bool> {
        // Out of range -> nothing to delete.
        if id >= self.storage.read().len() {
            return Ok(false);
        }
        // Mark deleted; bail out if it was already deleted.
        if !self.deleted.write().insert(id) {
            return Ok(false);
        }
        // Clear metadata so a stale payload can't match a filter or be returned.
        {
            let mut meta = self.metadata.write();
            if let Some(slot) = meta.get_mut(id) {
                *slot = Metadata::default();
            }
        }
        // Remove from disk (the in-memory slot survives until the next reopen).
        #[cfg(any(feature = "persistence-fjall", feature = "persistence-sled", feature = "persistence-rocksdb"))]
        {
            if let Some(backend) = &self.backend {
                backend.delete(&Self::rec_key(id))?;
            }
        }
        Ok(true)
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

        // Over-fetch up to the full search beam so soft-deletes and metadata
        // filters still leave enough live candidates to return k results.
        let fetch_k = ef_search.max(k);

        // Search HNSW index (uses E8 codec for distance; H4 decoded via storage on retrieval)
        let candidates = {
            let index = self.index.read();
            let storage = self.storage.read();
            index.search(
                &processed_query,
                fetch_k,
                ef_search,
                &storage,
                self.e8_codec.as_ref(),
            )?
        };

        // Apply deletes + filter and collect results
        let metadata = self.metadata.read();
        let deleted = self.deleted.read();
        let mut results: Vec<Hit> = candidates
            .into_iter()
            .filter_map(|(id, score)| {
                // Skip soft-deleted vectors.
                if deleted.contains(&id) {
                    return None;
                }

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

    /// Search many query vectors at once, returning one result list per query.
    ///
    /// Queries are run in parallel (Rayon). Equivalent to calling `search` in a
    /// loop, but uses all cores — useful for batched retrieval (e.g. a
    /// LangChain/LlamaIndex retriever issuing several queries).
    #[cfg(feature = "async")]
    pub async fn search_many(
        &self,
        queries: &[Vec<f32>],
        k: usize,
        ef_search: usize,
        filter: Option<FilterExpr>,
    ) -> Result<Vec<Vec<Hit>>> {
        self.search_many_internal(queries, k, ef_search, filter)
    }

    /// Internal batch search (public for Python bindings).
    pub fn search_many_internal(
        &self,
        queries: &[Vec<f32>],
        k: usize,
        ef_search: usize,
        filter: Option<FilterExpr>,
    ) -> Result<Vec<Vec<Hit>>> {
        use rayon::prelude::*;
        queries
            .par_iter()
            .map(|q| self.search_internal(q, k, ef_search, filter.clone()))
            .collect()
    }

    /// Get the metadata payload for a live id (`None` if out of range or deleted).
    pub fn payload(&self, id: usize) -> Option<Metadata> {
        if self.deleted.read().contains(&id) {
            return None;
        }
        self.metadata.read().get(id).cloned()
    }

    /// Whether `id` refers to a live (non-deleted) vector.
    pub fn contains_id(&self, id: usize) -> bool {
        id < self.storage.read().len() && !self.deleted.read().contains(&id)
    }

    /// All live `(id, payload)` pairs.
    ///
    /// Lets callers rebuild an external key→id index (e.g. a document-id map for
    /// a LangChain/LlamaIndex adapter) after reopening a persisted store, since
    /// ids are stable across reload.
    pub fn entries(&self) -> Vec<(usize, Metadata)> {
        let metadata = self.metadata.read();
        let deleted = self.deleted.read();
        metadata
            .iter()
            .enumerate()
            .filter(|(id, _)| !deleted.contains(id))
            .map(|(id, m)| (id, m.clone()))
            .collect()
    }

    /// Number of live (non-deleted) vectors (synchronous; for Python bindings).
    pub fn live_count(&self) -> usize {
        self.storage
            .read()
            .len()
            .saturating_sub(self.deleted.read().len())
    }

    /// Get the number of live (non-deleted) vectors in the database
    #[cfg(feature = "async")]
    pub async fn len(&self) -> usize {
        self.live_count()
    }

    /// Check if the database has no live vectors
    #[cfg(feature = "async")]
    pub async fn is_empty(&self) -> bool {
        self.len().await == 0
    }

    /// Clear all vectors and metadata
    #[cfg(feature = "async")]
    pub async fn clear(&mut self) -> Result<()> {
        self.clear_sync()
    }

    /// Synchronous clear (public for Python bindings).
    pub fn clear_sync(&mut self) -> Result<()> {
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
        {
            let mut deleted = self.deleted.write();
            deleted.clear();
        }

        // Remove persisted vector records (the header is kept so the store's
        // configuration survives an empty reopen).
        #[cfg(any(feature = "persistence-fjall", feature = "persistence-sled", feature = "persistence-rocksdb"))]
        self.clear_backend()?;

        Ok(())
    }

    /// Flush changes to disk (if persistence enabled)
    #[cfg(all(feature = "async", any(feature = "persistence-fjall", feature = "persistence-sled", feature = "persistence-rocksdb")))]
    pub async fn flush(&mut self) -> Result<()> {
        // Record the current high-water mark so that ids (including trailing
        // deleted ones) stay stable across a flush + reopen.
        self.persist_header()?;
        if let Some(ref backend) = self.backend {
            backend.flush()?;
        }
        Ok(())
    }

    /// Get current quantization mode
    pub fn quantization(&self) -> &Quantization {
        &self.quantization
    }

    /// Set quantization mode (re-quantizes all existing vectors)
    #[cfg(feature = "async")]
    pub async fn set_quantization(&mut self, quant: Quantization) -> Result<()> {
        self.e8_codec = match &quant {
            Quantization::E8 { bits_per_block, use_hadamard, random_seed } =>
                Some(E8Codec::new(self.dimension, *bits_per_block, *use_hadamard, *random_seed)),
            _ => None,
        };
        self.h4_codec = match &quant {
            Quantization::H4 { use_hadamard, random_seed } =>
                Some(H4Codec::new(self.dimension, *use_hadamard, *random_seed)),
            _ => None,
        };
        self.quantization = quant.clone();

        // Keep the index's H4 codec in sync with the new quantization so its
        // distance computations decode H4 vectors correctly.
        self.index.write().set_h4_codec(self.h4_codec.clone());

        // Re-quantize existing vectors (scoped so the write lock is released
        // before any persistence re-encode below re-reads storage).
        {
            let mut storage = self.storage.write();
            storage.set_quantization(quant, self.e8_codec.as_ref(), self.h4_codec.as_ref())?;
        }

        // The on-disk encoding changed for every vector — rewrite header + records.
        #[cfg(any(feature = "persistence-fjall", feature = "persistence-sled", feature = "persistence-rocksdb"))]
        self.persist_all()?;

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

// =============================================================================
// Persistence: save/load vectors + metadata + index through the backend
// =============================================================================

#[cfg(any(feature = "persistence-fjall", feature = "persistence-sled", feature = "persistence-rocksdb"))]
impl EmbedVec {
    /// Backend key for a record id (zero-padded so prefix scans stay id-ordered).
    fn rec_key(id: usize) -> Vec<u8> {
        format!("rec:{:020}", id).into_bytes()
    }

    /// Serialize the record (stored vector + metadata) for a given id.
    fn encode_record(&self, id: usize) -> Result<Vec<u8>> {
        let stored = self
            .storage
            .read()
            .get_stored(id)
            .cloned()
            .ok_or(EmbedVecError::VectorNotFound(id))?;
        let meta = self.metadata.read().get(id).cloned().unwrap_or_default();
        let record = PersistedRecord { stored, meta };
        serde_json::to_vec(&record)
            .map_err(|e| EmbedVecError::SerializationError(e.to_string()))
    }

    /// Write the self-describing header for the current configuration.
    fn persist_header(&self) -> Result<()> {
        let backend = match &self.backend {
            Some(b) => b,
            None => return Ok(()),
        };
        let header = PersistHeader {
            version: PERSIST_VERSION,
            dimension: self.dimension,
            distance: self.distance,
            quantization: self.quantization.clone(),
            m: self.index.read().m(),
            ef_construction: self.index.read().ef_construction(),
            high_water_mark: self.storage.read().len(),
        };
        let bytes = serde_json::to_vec(&header)
            .map_err(|e| EmbedVecError::SerializationError(e.to_string()))?;
        backend.set(PERSIST_HEADER_KEY, &bytes)
    }

    /// Persist a single record.
    fn persist_record(&self, id: usize) -> Result<()> {
        let backend = match &self.backend {
            Some(b) => b,
            None => return Ok(()),
        };
        let value = self.encode_record(id)?;
        backend.set(&Self::rec_key(id), &value)
    }

    /// Persist many records in one atomic batch.
    fn persist_records(&self, ids: &[usize]) -> Result<()> {
        let backend = match &self.backend {
            Some(b) => b,
            None => return Ok(()),
        };
        if ids.is_empty() {
            return Ok(());
        }
        let mut batch: Vec<(Vec<u8>, Vec<u8>)> = Vec::with_capacity(ids.len());
        for &id in ids {
            batch.push((Self::rec_key(id), self.encode_record(id)?));
        }
        backend.set_batch(&batch)
    }

    /// Rewrite the header and every record (used after a quantization change).
    fn persist_all(&self) -> Result<()> {
        if self.backend.is_none() {
            return Ok(());
        }
        self.persist_header()?;
        let n = self.storage.read().len();
        let ids: Vec<usize> = (0..n).collect();
        self.persist_records(&ids)
    }

    /// Delete all persisted vector records (keeps the header).
    fn clear_backend(&self) -> Result<()> {
        let backend = match &self.backend {
            Some(b) => b,
            None => return Ok(()),
        };
        let keys: Vec<Vec<u8>> = backend
            .scan_prefix(PERSIST_REC_PREFIX)?
            .into_iter()
            .map(|(k, _)| k)
            .collect();
        for key in keys {
            backend.delete(&key)?;
        }
        Ok(())
    }

    /// Reload persisted vectors on open.
    ///
    /// If a header exists, its configuration is adopted (on-disk vectors are
    /// encoded against it), all records are loaded, and the HNSW index is
    /// rebuilt. If no header exists, the store is treated as fresh and the
    /// current configuration is recorded for consistent future reopens.
    ///
    /// Gaps in the id sequence (from deletes) are refilled with tombstones, and
    /// trailing deletes are restored up to the persisted high-water mark, so
    /// surviving ids stay stable and deleted ids are never reused.
    fn load_from_backend(&mut self) -> Result<()> {
        let header_bytes = match &self.backend {
            Some(backend) => backend.get(PERSIST_HEADER_KEY)?,
            None => return Ok(()),
        };

        let header: PersistHeader = match header_bytes {
            Some(bytes) => serde_json::from_slice(&bytes)
                .map_err(|e| EmbedVecError::SerializationError(e.to_string()))?,
            // Fresh store: record the configuration so reopening is consistent.
            None => return self.persist_header(),
        };

        if header.dimension != self.dimension {
            return Err(EmbedVecError::DimensionMismatch {
                expected: header.dimension,
                got: self.dimension,
            });
        }

        // Adopt the persisted configuration — on-disk vectors are encoded against it.
        self.distance = header.distance;
        self.quantization = header.quantization.clone();
        self.e8_codec = match &self.quantization {
            Quantization::E8 { bits_per_block, use_hadamard, random_seed } =>
                Some(E8Codec::new(self.dimension, *bits_per_block, *use_hadamard, *random_seed)),
            _ => None,
        };
        self.h4_codec = match &self.quantization {
            Quantization::H4 { use_hadamard, random_seed } =>
                Some(H4Codec::new(self.dimension, *use_hadamard, *random_seed)),
            _ => None,
        };
        *self.storage.write() = VectorStorage::new(self.dimension, self.quantization.clone());
        let mut rebuilt = HnswIndex::new(header.m, header.ef_construction, self.distance);
        rebuilt.set_h4_codec(self.h4_codec.clone());
        *self.index.write() = rebuilt;
        self.metadata.write().clear();
        self.deleted.write().clear();

        // Records come back in ascending id order (zero-padded keys). Deleted
        // ids appear as gaps; fill those slots with tombstones so surviving ids
        // stay stable (id == position). Tombstones never enter the HNSW graph.
        let records = match &self.backend {
            Some(backend) => backend.scan_prefix(PERSIST_REC_PREFIX)?,
            None => return Ok(()),
        };

        let mut deleted_ids: HashSet<usize> = HashSet::new();
        {
            let mut storage = self.storage.write();
            let mut metadata = self.metadata.write();
            let mut index = self.index.write();
            let mut next_id = 0usize;

            for (key, value) in records {
                let id: usize = std::str::from_utf8(&key[PERSIST_REC_PREFIX.len()..])
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .ok_or_else(|| {
                        EmbedVecError::PersistenceError(format!(
                            "invalid record key: {}",
                            String::from_utf8_lossy(&key)
                        ))
                    })?;

                // Fill any gap before this id with tombstones (deleted slots).
                while next_id < id {
                    storage.push_stored(crate::storage::StoredVector::Tombstone);
                    metadata.push(Metadata::default());
                    deleted_ids.insert(next_id);
                    next_id += 1;
                }

                let record: PersistedRecord = serde_json::from_slice(&value)
                    .map_err(|e| EmbedVecError::SerializationError(e.to_string()))?;
                // Decode for the index before moving the stored form into storage.
                let decoded = record
                    .stored
                    .to_f32(self.e8_codec.as_ref(), self.h4_codec.as_ref());
                storage.push_stored(record.stored);
                metadata.push(record.meta);
                index.insert(id, &decoded, &*storage, self.e8_codec.as_ref())?;
                next_id += 1;
            }

            // Recreate trailing deleted slots up to the recorded high-water mark.
            while next_id < header.high_water_mark {
                storage.push_stored(crate::storage::StoredVector::Tombstone);
                metadata.push(Metadata::default());
                deleted_ids.insert(next_id);
                next_id += 1;
            }
        }

        *self.deleted.write() = deleted_ids;

        Ok(())
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

    #[tokio::test]
    async fn test_h4_search_recall_multi() {
        // Regression test for the H4 codec not being threaded into the HNSW
        // index: H4 vectors used to decode to zeros during distance computation,
        // so every candidate was equidistant and search returned garbage. With
        // distinct vectors, each should now retrieve itself.
        let dim = 128;
        let n = 16;
        let mut db = EmbedVec::builder()
            .dimension(dim)
            .metric(Distance::Cosine)
            .quantization(Quantization::h4_default())
            .build()
            .await
            .unwrap();

        // Well-separated pseudo-random vectors.
        let mut seed = 0x1234_5678u64;
        let mut rng = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed as f32 / u64::MAX as f32) * 2.0 - 1.0
        };
        let mut vectors = Vec::new();
        for i in 0..n {
            let v: Vec<f32> = (0..dim).map(|_| rng()).collect();
            db.add(&v, serde_json::json!({ "i": i })).await.unwrap();
            vectors.push(v);
        }

        // Each vector should retrieve itself as the nearest neighbor.
        let mut self_hits = 0;
        for (i, v) in vectors.iter().enumerate() {
            let hits = db.search(v, 1, 64, None).await.unwrap();
            if hits[0].id == i {
                self_hits += 1;
            }
        }
        assert!(
            self_hits >= n - 1,
            "H4 self-recall too low: {self_hits}/{n} (index not decoding H4?)"
        );

        // Anti-garbage: distinct queries return distinct nearest neighbors.
        let h0 = db.search(&vectors[0], 1, 64, None).await.unwrap()[0].id;
        let h_last = db.search(&vectors[n - 1], 1, 64, None).await.unwrap()[0].id;
        assert_ne!(h0, h_last, "distinct H4 queries returned the same top hit");
    }

    #[tokio::test]
    async fn test_search_many_and_entries() {
        let mut db = EmbedVec::new(4, Distance::Cosine, 16, 100).await.unwrap();
        let a = db.add(&[1.0, 0.0, 0.0, 0.0], serde_json::json!({"doc": "a"})).await.unwrap();
        let b = db.add(&[0.0, 1.0, 0.0, 0.0], serde_json::json!({"doc": "b"})).await.unwrap();
        let _c = db.add(&[0.0, 0.0, 1.0, 0.0], serde_json::json!({"doc": "c"})).await.unwrap();

        // Batch search: each query returns its own ranked list.
        let queries = vec![vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0]];
        let results = db.search_many(&queries, 1, 50, None).await.unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0][0].id, a);
        assert_eq!(results[1][0].id, b);

        // entries() / payload() expose live (id, payload) for external id maps.
        let entries = db.entries();
        assert_eq!(entries.len(), 3);
        assert_eq!(db.payload(a).unwrap()["doc"], "a");
        assert!(db.contains_id(b));

        // After delete, entries/payload reflect liveness.
        db.delete(b).await.unwrap();
        assert_eq!(db.entries().len(), 2);
        assert!(db.payload(b).is_none());
        assert!(!db.contains_id(b));
    }

    #[tokio::test]
    async fn test_delete_in_memory() {
        let mut db = EmbedVec::new(4, Distance::Cosine, 16, 100).await.unwrap();
        let a = db.add(&[1.0, 0.0, 0.0, 0.0], serde_json::json!({"v": "a"})).await.unwrap();
        let b = db.add(&[0.0, 1.0, 0.0, 0.0], serde_json::json!({"v": "b"})).await.unwrap();
        let c = db.add(&[0.0, 0.0, 1.0, 0.0], serde_json::json!({"v": "c"})).await.unwrap();
        assert_eq!(db.len().await, 3);

        // Delete b.
        assert!(db.delete(b).await.unwrap());
        assert_eq!(db.len().await, 2);

        // Deleting again, or an out-of-range id, returns false.
        assert!(!db.delete(b).await.unwrap());
        assert!(!db.delete(999).await.unwrap());

        // b is excluded from search; a and c are still found.
        let hits = db.search(&[0.0, 1.0, 0.0, 0.0], 3, 50, None).await.unwrap();
        assert!(hits.iter().all(|h| h.id != b));
        assert_eq!(db.search(&[1.0, 0.0, 0.0, 0.0], 1, 50, None).await.unwrap()[0].id, a);
        assert_eq!(db.search(&[0.0, 0.0, 1.0, 0.0], 1, 50, None).await.unwrap()[0].id, c);

        // A new add gets a fresh id — b's id is not reused.
        let d = db.add(&[0.0, 0.0, 0.0, 1.0], serde_json::json!({"v": "d"})).await.unwrap();
        assert_eq!(d, 3);
        assert_ne!(d, b);
        assert_eq!(db.len().await, 3);
    }

    #[tokio::test]
    async fn test_h4_quantization_end_to_end() {
        let mut db = EmbedVec::builder()
            .dimension(8)
            .metric(Distance::Cosine)
            .quantization(Quantization::h4_default())
            .build()
            .await
            .unwrap();

        let id = db
            .add(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], serde_json::json!({"lattice": "h4"}))
            .await
            .unwrap();
        assert_eq!(id, 0);

        let results = db
            .search(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 1, 50, None)
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, 0);
    }
}
