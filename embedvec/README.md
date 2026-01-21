# embedvec — High-Performance Embedded Vector Database

[![crates.io](https://img.shields.io/crates/v/embedvec.svg)](https://crates.io/crates/embedvec)
[![docs.rs](https://docs.rs/embedvec/badge.svg)](https://docs.rs/embedvec)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Fast, lightweight, in-process vector database** with HNSW indexing, SIMD-accelerated distance calculations, metadata filtering, E8 lattice quantization, and optional persistence via Sled or RocksDB.

## Why embedvec?

- **Pure Rust** — No C++ dependencies (unless using RocksDB), safe and portable
- **Blazing Fast** — AVX2/FMA SIMD acceleration, optimized HNSW with O(1) lookups
- **Memory Efficient** — E8 quantization provides 4-6× compression with <5% recall loss
- **Flexible Persistence** — Choose between Sled (pure Rust) or RocksDB (high performance)
- **Production Ready** — Async API, metadata filtering, batch operations

## Benchmarks

**768-dimensional vectors, 10k dataset, AVX2 enabled:**

| Operation | Time | Throughput |
|-----------|------|------------|
| **Search (ef=32)** | 3.0 ms | 3,300 queries/sec |
| **Search (ef=64)** | 4.9 ms | 2,000 queries/sec |
| **Search (ef=128)** | 16.1 ms | 620 queries/sec |
| **Search (ef=256)** | 23.2 ms | 430 queries/sec |
| **Insert (768-dim)** | 25.5 ms/100 | 3,900 vectors/sec |
| **Distance (cosine)** | 122 ns/pair | 8.2M ops/sec |
| **Distance (euclidean)** | 108 ns/pair | 9.3M ops/sec |
| **Distance (dot product)** | 91 ns/pair | 11M ops/sec |

*Run `cargo bench` to reproduce on your hardware.*

## Core Features

| Feature | Description |
|---------|-------------|
| **HNSW Indexing** | Hierarchical Navigable Small World graph for O(log n) ANN search |
| **SIMD Distance** | AVX2/FMA accelerated cosine, euclidean, dot product |
| **E8 Quantization** | Lattice-based compression (4-6× memory reduction) |
| **Metadata Filtering** | Composable filters: eq, gt, lt, contains, AND/OR/NOT |
| **Dual Persistence** | Sled (pure Rust) or RocksDB (high performance) |
| **Async API** | Tokio-compatible async operations |
| **PyO3 Bindings** | First-class Python support with numpy interop |
| **WASM Support** | Feature-gated for browser/edge deployment |

## Quick Start — Rust

```toml
[dependencies]
embedvec = "0.5"
tokio = { version = "1.0", features = ["rt-multi-thread", "macros"] }
serde_json = "1.0"
```

```rust
use embedvec::{Distance, EmbedVec, FilterExpr, Quantization};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create in-memory index with E8 quantization
    let mut db = EmbedVec::builder()
        .dimension(768)
        .metric(Distance::Cosine)
        .m(32)                              // HNSW connections per layer
        .ef_construction(200)               // Build-time beam width
        .quantization(Quantization::e8_default())  // 4-6× memory savings
        .build()
        .await?;

    // Add vectors with metadata
    let vectors = vec![
        vec![0.1; 768],
        vec![0.2; 768],
    ];
    let payloads = vec![
        serde_json::json!({"doc_id": "123", "category": "finance", "timestamp": 1737400000}),
        serde_json::json!({"doc_id": "456", "category": "tech", "timestamp": 1737500000}),
    ];

    db.add_many(&vectors, payloads).await?;

    // Search with metadata filter
    let filter = FilterExpr::eq("category", "finance")
        .and(FilterExpr::gt("timestamp", 1730000000));

    let results = db.search(
        &vec![0.15; 768],  // query vector
        10,                 // k
        128,                // ef_search
        Some(filter)
    ).await?;

    for hit in results {
        println!("id: {}, score: {:.4}, payload: {:?}", hit.id, hit.score, hit.payload);
    }

    Ok(())
}
```

## Quick Start — Python

```bash
pip install embedvec-py
```

```python
import embedvec_py
import numpy as np

# Create database with E8 quantization
db = embedvec_py.EmbedVec(
    dim=768,
    metric="cosine",
    m=32,
    ef_construction=200,
    quantization="e8-10bit",  # or None, "e8-8bit", "e8-12bit"
    persist_path=None,         # or "/tmp/embedvec.db"
)

# Add vectors (numpy array or list-of-lists)
vectors = np.random.randn(50000, 768).tolist()
payloads = [{"doc_id": str(i), "tag": "news" if i % 3 == 0 else "blog"} 
            for i in range(50000)]

db.add_many(vectors, payloads)

# Search with filter
query = np.random.randn(768).tolist()
hits = db.search(
    query_vector=query,
    k=10,
    ef_search=128,
    filter={"tag": "news"}  # simple exact-match shorthand
)

for hit in hits:
    print(f"score: {hit['score']:.4f}  id: {hit['id']}  {hit['payload']}")
```

## API Reference

### EmbedVec Builder

```rust
EmbedVec::builder()
    .dimension(768)                    // Vector dimension (required)
    .metric(Distance::Cosine)          // Distance metric
    .m(32)                             // HNSW M parameter
    .ef_construction(200)              // HNSW build parameter
    .quantization(Quantization::None)  // Or E8 for compression
    .persistence("path/to/db")         // Optional disk persistence
    .build()
    .await?;
```

### Core Operations

| Method | Description |
|--------|-------------|
| `add(vector, payload)` | Add single vector with metadata |
| `add_many(vectors, payloads)` | Batch add vectors |
| `search(query, k, ef_search, filter)` | Find k nearest neighbors |
| `len()` | Number of vectors |
| `clear()` | Remove all vectors |
| `flush()` | Persist to disk (if enabled) |

### FilterExpr — Composable Filters

```rust
// Equality
FilterExpr::eq("category", "finance")

// Comparisons
FilterExpr::gt("timestamp", 1730000000)
FilterExpr::gte("score", 0.5)
FilterExpr::lt("price", 100)
FilterExpr::lte("count", 10)

// String operations
FilterExpr::contains("name", "test")
FilterExpr::starts_with("path", "/api")

// Membership
FilterExpr::in_values("status", vec!["active", "pending"])

// Existence
FilterExpr::exists("optional_field")

// Boolean composition
FilterExpr::eq("a", 1)
    .and(FilterExpr::eq("b", 2))
    .or(FilterExpr::not(FilterExpr::eq("c", 3)))
```

### Quantization Modes

| Mode | Bits/Dim | Memory/Vector (768d) | Recall@10 |
|------|----------|----------------------|-----------|
| `None` | 32 | ~3.1 KB | 100% |
| `E8 8-bit` | ~1.0 | ~170 B | 92–97% |
| `E8 10-bit` | ~1.25 | ~220 B | 96–99% |
| `E8 12-bit` | ~1.5 | ~280 B | 98–99% |

```rust
// No quantization (full f32)
Quantization::None

// E8 with Hadamard preprocessing (recommended)
Quantization::E8 {
    bits_per_block: 10,
    use_hadamard: true,
    random_seed: 0xcafef00d,
}

// Convenience constructor
Quantization::e8_default()  // 10-bit with Hadamard
```

## E8 Lattice Quantization

embedvec implements state-of-the-art E8 lattice quantization based on QuIP#/NestQuant/QTIP research (2024-2025):

1. **Hadamard Preprocessing**: Fast Walsh-Hadamard transform + random signs makes coordinates more Gaussian/i.i.d.
2. **Block-wise Quantization**: Split vectors into 8D blocks, quantize each to nearest E8 lattice point
3. **Asymmetric Search**: Query remains FP32, database vectors decoded on-the-fly during HNSW traversal
4. **Compact Storage**: ~2-2.5 bits per dimension effective

### Why E8?

The E8 lattice has exceptional packing density in 8 dimensions, providing better rate-distortion than scalar quantization or product quantization for normalized embeddings typical in LLM/RAG applications.

## Performance

### Measured Benchmarks (768-dim, 10k vectors, AVX2)

| Operation | Time | Throughput |
|-----------|------|------------|
| **Search (ef=32)** | 3.0 ms | 3,300 queries/sec |
| **Search (ef=64)** | 4.9 ms | 2,000 queries/sec |
| **Search (ef=128)** | 16.1 ms | 620 queries/sec |
| **Search (ef=256)** | 23.2 ms | 430 queries/sec |
| **Insert (768-dim)** | 25.5 ms/100 | 3,900 vectors/sec |
| **Distance (cosine)** | 122 ns/pair | 8.2M ops/sec |
| **Distance (euclidean)** | 108 ns/pair | 9.3M ops/sec |
| **Distance (dot product)** | 91 ns/pair | 11M ops/sec |

### Projected Performance at Scale

| Operation | ~1M vectors | ~10M vectors | Notes |
|-----------|-------------|--------------|-------|
| Query (k=10, ef=128) | 0.4–1.2 ms | 1–4 ms | Cosine, no filter |
| Query + filter | 0.6–2.5 ms | 2–8 ms | Depends on selectivity |
| Memory (FP32) | ~3.1 GB | ~31 GB | Full precision |
| Memory (E8-10bit) | ~0.5 GB | ~5 GB | 4-6× reduction |

## Feature Flags

```toml
[dependencies]
embedvec = { version = "0.5", features = ["persistence-sled", "async"] }
```

| Feature | Description | Default |
|---------|-------------|---------|
| `persistence-sled` | On-disk storage via Sled (pure Rust) | ✓ |
| `persistence-rocksdb` | On-disk storage via RocksDB (higher perf) | ✗ |
| `async` | Tokio async API | ✓ |
| `python` | PyO3 bindings | ✗ |
| `simd` | SIMD distance optimizations | ✗ |
| `wasm` | WebAssembly support | ✗ |

## Persistence Backends

embedvec supports two persistence backends:

### Sled (Default)
Pure Rust embedded database. Good default for most use cases.

```rust
use embedvec::{EmbedVec, Distance, BackendConfig, BackendType};

// Simple path-based persistence (uses Sled)
let db = EmbedVec::with_persistence("/path/to/db", 768, Distance::Cosine, 32, 200).await?;

// Or via builder
let db = EmbedVec::builder()
    .dimension(768)
    .persistence("/path/to/db")
    .build()
    .await?;
```

### RocksDB (Optional)
Higher performance LSM-tree database. Better for write-heavy workloads and large datasets.

```toml
[dependencies]
embedvec = { version = "0.5", features = ["persistence-rocksdb", "async"] }
```

```rust
use embedvec::{EmbedVec, Distance, BackendConfig, BackendType};

// Configure RocksDB backend
let config = BackendConfig::new("/path/to/db")
    .backend(BackendType::RocksDb)
    .cache_size(256 * 1024 * 1024);  // 256MB cache

let db = EmbedVec::with_backend(config, 768, Distance::Cosine, 32, 200).await?;
```

## Testing

```bash
# Run all tests
cargo test

# Run with specific features
cargo test --features "persistence"

# Run benchmarks
cargo bench
```

## Benchmarking

```bash
# Install criterion
cargo install cargo-criterion

# Run benchmarks
cargo criterion

# Memory profiling (requires jemalloc)
cargo bench --features "jemalloc"
```

## Roadmap

- **v0.5** (current): E8 quantization stable + persistence
- **v0.6**: Binary/PQ fallback, delete support, batch queries
- **v0.7**: LangChain/LlamaIndex official integration
- **Future**: Hybrid sparse-dense, full-text + vector

## License

MIT OR Apache-2.0

## Contributing

Contributions welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) before submitting PRs.

## Acknowledgments

- HNSW algorithm: Malkov & Yashunin (2016)
- E8 quantization: Inspired by QuIP#, NestQuant, QTIP (2024-2025)
- Rust ecosystem: serde, tokio, pyo3, sled

---

**embedvec** — The "SQLite of vector search" for Rust-first teams in 2026.
