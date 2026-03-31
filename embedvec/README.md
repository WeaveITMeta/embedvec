# embedvec — High-Performance Embedded Vector Database

[![crates.io](https://img.shields.io/crates/v/embedvec.svg)](https://crates.io/crates/embedvec)
[![docs.rs](https://docs.rs/embedvec/badge.svg)](https://docs.rs/embedvec)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**The fastest pure-Rust vector database** — HNSW indexing, SIMD acceleration, E8 and H4 lattice quantization, and flexible persistence (Sled, RocksDB, or PostgreSQL/pgvector).

---

## Why embedvec Over the Competition?

| Feature | embedvec | Qdrant | Milvus | Pinecone | pgvector |
|---------|----------|--------|--------|----------|----------|
| **Deployment** | Embedded (in-process) | Server | Server | Cloud-only | PostgreSQL extension |
| **Language** | Pure Rust | Rust | Go/C++ | Proprietary | C |
| **Latency** | <1ms p99 | 2-10ms | 5-20ms | 10-50ms | 2-5ms |
| **Memory (1M 768d)** | ~196MB (H4) / ~120MB (E8) | ~3GB | ~3GB | N/A | ~3GB |
| **Zero-copy** | ✓ | ✗ | ✗ | ✗ | ✗ |
| **SIMD** | AVX2/FMA | AVX2 | AVX2 | Unknown | ✗ |
| **Quantization** | E8 + H4 lattice (SOTA) | Scalar/PQ | PQ/SQ | Unknown | ✗ |
| **Python bindings** | ✓ (PyO3) | ✓ | ✓ | ✓ | ✓ (psycopg) |
| **WASM support** | ✓ | ✗ | ✗ | ✗ | ✗ |

### Key Advantages

1. **10-100× Lower Latency** — No network round-trips. embedvec runs in your process. Sub-millisecond queries are the norm, not the exception.

2. **Up to 16× Less Memory** — E8 and H4 lattice quantization (from QuIP#/QTIP research) achieve 1.25–1.73 bits/dimension with <5% recall loss. Store 1M 768-dim vectors in ~196 MB instead of 3 GB.

3. **No Infrastructure** — No Docker, no Kubernetes, no managed service bills. Just `cargo add embedvec`. Perfect for edge devices, mobile, WASM, and serverless.

4. **Scale When Ready** — Start embedded, then seamlessly migrate to PostgreSQL/pgvector for distributed deployments without changing your code.

5. **True Rust Safety** — No unsafe FFI, no C++ dependencies (unless you opt into RocksDB). Memory-safe, thread-safe, and panic-free.

### When to Use embedvec

| Use Case | embedvec | Server DB |
|----------|----------|-----------|
| RAG/LLM apps with <10M vectors | ✓ Best | Overkill |
| Edge/mobile/WASM deployment | ✓ Only option | ✗ |
| Prototype → production path | ✓ Same code | Rewrite needed |
| Multi-tenant SaaS | Consider | ✓ Better |
| >100M vectors | Consider pgvector | ✓ Better |

---

## Why embedvec?

- **Pure Rust** — No C++ dependencies (unless using RocksDB/pgvector), safe and portable
- **Blazing Fast** — AVX2/FMA SIMD acceleration, optimized HNSW with O(1) lookups
- **Memory Efficient** — H4 (~15.7×) and E8 (~24.8×) quantization with <5% recall loss
- **Two Lattice Modes** — E8 (8D, 240 roots) for maximum compression; H4 (4D, 600-cell) for fast decoding
- **Flexible Persistence** — Sled (pure Rust), RocksDB (high perf), or PostgreSQL/pgvector (distributed)
- **Production Ready** — Async API, metadata filtering, batch operations

---

## Benchmarks

All measurements on 768-dimensional vectors. Run `cargo bench -- lattice` to reproduce.

### Lattice Quantization Comparison (768-dim, 100 vectors per batch)

| Metric | None (raw f32) | H4 (600-cell) | E8 (D8 lattice) |
|--------|---------------|----------------|-----------------|
| **Encode / 100 vectors** | 15.3 µs | 7.26 ms | 3.29 ms |
| **Decode / 100 vectors** | 17.5 µs | 249 µs | 1.10 ms |
| **Insert / 100 vectors** | 32.7 ms | 36.2 ms (+11%) | 905 ms (+27×) |
| **Search / 10 queries (ef=64, 10k DB)** | 10.3 ms | 0.69 ms | 133 ms |
| **Bytes / vector (768-dim)** | 3,072 B | **196 B** | **124 B** |
| **Compression ratio** | 1× | **15.7×** | **24.8×** |
| **Bits / dimension** | 32 | ~1.73 | ~1.25 |

> **H4 search is fast** because HNSW indexes the raw float vector at insert time; the quantized H4 representation is used for storage only. E8 search decodes each candidate during HNSW graph traversal, adding decode overhead per distance call.

### Core Operations (768-dim, 10k dataset, AVX2)

| Operation | Time | Throughput |
|-----------|------|------------|
| **Search (ef=32)** | 3.0 ms | 3,300 queries/sec |
| **Search (ef=64)** | 4.9 ms | 2,000 queries/sec |
| **Search (ef=128)** | 16.1 ms | 620 queries/sec |
| **Search (ef=256)** | 23.2 ms | 430 queries/sec |
| **Insert (768-dim, raw)** | 32.7 ms/100 | 3,060 vectors/sec |
| **Distance (cosine)** | 122 ns/pair | 8.2M ops/sec |
| **Distance (euclidean)** | 108 ns/pair | 9.3M ops/sec |
| **Distance (dot product)** | 91 ns/pair | 11M ops/sec |

### Memory Usage at Scale (768-dim vectors)

| Mode | Bytes/Vector | 100k Vectors | 1M Vectors | Compression |
|------|-------------|-------------|------------|-------------|
| Raw f32 | 3,072 B | ~307 MB | ~3.07 GB | 1× |
| **H4** | **196 B** | **~19.6 MB** | **~196 MB** | **15.7×** |
| **E8 10-bit** | **124 B** | **~12.4 MB** | **~124 MB** | **24.8×** |

---

## Core Features

| Feature | Description |
|---------|-------------|
| **HNSW Indexing** | Hierarchical Navigable Small World graph for O(log n) ANN search |
| **SIMD Distance** | AVX2/FMA accelerated cosine, euclidean, dot product |
| **E8 Quantization** | 8D D8∪D8+½ lattice, 240 roots, ~1.25 bits/dim, 24.8× compression |
| **H4 Quantization** | 4D 600-cell polytope, 120 vertices, ~1.73 bits/dim, 15.7× compression |
| **Metadata Filtering** | Composable filters: eq, gt, lt, contains, AND/OR/NOT |
| **Triple Persistence** | Sled (pure Rust), RocksDB (high perf), or pgvector (PostgreSQL) |
| **pgvector Integration** | Native PostgreSQL vector search with HNSW/IVFFlat indexes |
| **Async API** | Tokio-compatible async operations |
| **PyO3 Bindings** | First-class Python support with numpy interop |
| **WASM Support** | Feature-gated for browser/edge deployment |

---

## Quick Start — Rust

```toml
[dependencies]
embedvec = "0.6"
tokio = { version = "1.0", features = ["rt-multi-thread", "macros"] }
serde_json = "1.0"
```

```rust
use embedvec::{Distance, EmbedVec, FilterExpr, Quantization};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // H4: best balance of compression (15.7×) and fast decode
    let mut db = EmbedVec::builder()
        .dimension(768)
        .metric(Distance::Cosine)
        .m(32)
        .ef_construction(200)
        .quantization(Quantization::h4_default())  // 15.7× memory savings
        .build()
        .await?;

    // Or E8 for maximum compression (24.8×) at the cost of slower encode/search
    // .quantization(Quantization::e8_default())

    // Add vectors with metadata
    let vectors = vec![vec![0.1; 768], vec![0.2; 768]];
    let payloads = vec![
        serde_json::json!({"doc_id": "123", "category": "finance", "timestamp": 1737400000}),
        serde_json::json!({"doc_id": "456", "category": "tech",    "timestamp": 1737500000}),
    ];
    db.add_many(&vectors, payloads).await?;

    // Search with metadata filter
    let filter = FilterExpr::eq("category", "finance")
        .and(FilterExpr::gt("timestamp", 1730000000));

    let results = db.search(&vec![0.15; 768], 10, 128, Some(filter)).await?;

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

# Create database with H4 quantization (15.7× memory savings, fast decode)
db = embedvec_py.EmbedVec(
    dim=768,
    metric="cosine",
    m=32,
    ef_construction=200,
    quantization="h4",     # or None, "e8-10bit", "e8-8bit", "e8-12bit"
    persist_path=None,
)

vectors = np.random.randn(50000, 768).tolist()
payloads = [{"doc_id": str(i), "tag": "news" if i % 3 == 0 else "blog"}
            for i in range(50000)]
db.add_many(vectors, payloads)

query = np.random.randn(768).tolist()
hits = db.search(query_vector=query, k=10, ef_search=128, filter={"tag": "news"})

for hit in hits:
    print(f"score: {hit['score']:.4f}  id: {hit['id']}  {hit['payload']}")
```

---

## API Reference

### EmbedVec Builder

```rust
EmbedVec::builder()
    .dimension(768)                         // Vector dimension (required)
    .metric(Distance::Cosine)               // Distance metric
    .m(32)                                  // HNSW M parameter
    .ef_construction(200)                   // HNSW build parameter
    .quantization(Quantization::h4_default()) // None | h4_default() | e8_default()
    .persistence("path/to/db")             // Optional disk persistence
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
FilterExpr::eq("category", "finance")
FilterExpr::gt("timestamp", 1730000000)
FilterExpr::gte("score", 0.5)
FilterExpr::lt("price", 100)
FilterExpr::contains("name", "test")
FilterExpr::starts_with("path", "/api")
FilterExpr::in_values("status", vec!["active", "pending"])
FilterExpr::exists("optional_field")

// Boolean composition
FilterExpr::eq("a", 1)
    .and(FilterExpr::eq("b", 2))
    .or(FilterExpr::not(FilterExpr::eq("c", 3)))
```

---

## Quantization Reference

### Choosing a Mode

| Mode | Bits/Dim | Bytes/Vector (768d) | Encode Speed | Decode Speed | Best For |
|------|----------|----------------------|-------------|--------------|---------|
| `None` | 32 | 3,072 B | Instant | Instant | Highest accuracy, max RAM |
| `H4` | ~1.73 | 196 B | 72 µs/vec | 2.5 µs/vec | **Best balance** — fast decode, 15.7× compression |
| `E8 10-bit` | ~1.25 | 124 B | 33 µs/vec | 11 µs/vec | Maximum compression, slower search |

### H4 — 4D 600-Cell Lattice

```rust
// Default: Hadamard preprocessing, reproducible seed
Quantization::h4_default()

// Custom
Quantization::H4 {
    use_hadamard: true,
    random_seed: 0xdeadbeef,
}
```

The **H4 quantizer** maps each 4D block to the nearest vertex of the regular 600-cell polytope (120 vertices with icosahedral symmetry). Each block is stored as a single `u8` index.

- ~1.73 bits/dimension effective
- 15.7× compression vs raw f32 at 768 dimensions
- Fast decode: table lookup + 4D Hadamard inverse (~2.5 µs per vector)

### E8 — 8D D8 Lattice

```rust
// Default: 10-bit, Hadamard preprocessing
Quantization::e8_default()

// Custom bit-depth
Quantization::E8 {
    bits_per_block: 10,   // 8, 10, or 12
    use_hadamard: true,
    random_seed: 0xcafef00d,
}
```

The **E8 quantizer** uses the D8 ∪ (D8 + ½) double-cover decomposition to find the nearest E8 lattice point per 8D block. Achieves maximum compression density.

- ~1.25 bits/dimension effective
- 24.8× compression vs raw f32 at 768 dimensions
- Slower decode than H4 due to 8D parity reconstruction

---

## E8 and H4 Lattice Quantization

Both quantizers implement the same pipeline:

1. **Random Sign Preprocessing** — Multiply each coordinate by ±1 from a seeded PRNG
2. **Hadamard Transform** — Fast Walsh-Hadamard transform decorrelates coordinates
3. **Scale Normalization** — Global scale factor computed per vector
4. **Nearest Lattice Point** — Exhaustive search over roots (E8: 240, H4: 120)
5. **Compact Storage** — E8: u16 code + f32 scale; H4: u8 index per 4D block + f32 scale
6. **Asymmetric Search** — Query stays FP32; database decoded on-the-fly

Based on QuIP#/NestQuant/QTIP research (2024–2025).

---

## Performance

### Projected Performance at Scale

| Operation | ~1M vectors | ~10M vectors | Notes |
|-----------|-------------|--------------|-------|
| Query (k=10, ef=128) | 0.4–1.2 ms | 1–4 ms | Cosine, no filter |
| Query + filter | 0.6–2.5 ms | 2–8 ms | Depends on selectivity |
| Memory (None/f32) | ~3.1 GB | ~31 GB | Full precision |
| Memory (H4) | ~196 MB | ~1.96 GB | 15.7× reduction |
| Memory (E8 10-bit) | ~124 MB | ~1.24 GB | 24.8× reduction |

---

## Feature Flags

```toml
[dependencies]
embedvec = { version = "0.6", features = ["persistence-sled", "async"] }
```

| Feature | Description | Default |
|---------|-------------|---------|
| `persistence-sled` | On-disk storage via Sled (pure Rust) | ✓ |
| `persistence-rocksdb` | On-disk storage via RocksDB (higher perf) | ✗ |
| `persistence-pgvector` | PostgreSQL with native vector search | ✗ |
| `async` | Tokio async API | ✓ |
| `python` | PyO3 bindings | ✗ |
| `simd` | SIMD distance optimizations | ✗ |
| `wasm` | WebAssembly support | ✗ |

---

## Persistence Backends

### Sled (Default)
Pure Rust embedded database.

```rust
let db = EmbedVec::with_persistence("/path/to/db", 768, Distance::Cosine, 32, 200).await?;
```

### RocksDB (Optional)

```toml
embedvec = { version = "0.6", features = ["persistence-rocksdb", "async"] }
```

```rust
let config = BackendConfig::new("/path/to/db")
    .backend(BackendType::RocksDb)
    .cache_size(256 * 1024 * 1024);
let db = EmbedVec::with_backend(config, 768, Distance::Cosine, 32, 200).await?;
```

### pgvector (PostgreSQL)

```toml
embedvec = { version = "0.6", features = ["persistence-pgvector", "async"] }
```

```rust
let config = BackendConfig::pgvector("postgresql://user:pass@localhost/mydb", 768)
    .table_name("my_vectors")
    .index_type("hnsw");
let backend = PgVectorBackend::connect(&config).await?;
```

---

## Testing

```bash
cargo test

# Lattice comparison benchmarks only
cargo bench -- lattice

# Full benchmark suite
cargo bench
```

---

## Roadmap

- **v0.6** (current): H4 lattice quantization, E8 fixes, lattice benchmark suite
- **v0.7**: Delete support, batch queries, LangChain/LlamaIndex integration
- **Future**: Hybrid sparse-dense, full-text + vector, SIMD-accelerated lattice decode

---

## License

MIT OR Apache-2.0

## Contributing

Contributions welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) before submitting PRs.

## Acknowledgments

- HNSW algorithm: Malkov & Yashunin (2016)
- E8 quantization: Inspired by QuIP#, NestQuant, QTIP (2024–2025)
- H4 quantization: Regular 600-cell polytope (icosahedral symmetry in ℝ⁴)
- Rust ecosystem: serde, tokio, pyo3, sled

---

**embedvec** — The "SQLite of vector search" for Rust-first teams in 2026.
