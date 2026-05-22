# embedvec — High-Performance Embedded Vector Database

[![crates.io](https://img.shields.io/crates/v/embedvec.svg)](https://crates.io/crates/embedvec)
[![docs.rs](https://docs.rs/embedvec/badge.svg)](https://docs.rs/embedvec)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**The fastest pure-Rust vector database** — HNSW indexing, SIMD acceleration, E8 and H4 lattice quantization, and flexible persistence (Fjall by default, plus Sled, RocksDB, or PostgreSQL/pgvector).

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

2. **Up to 24.8× Smaller Vectors** — E8 and H4 lattice quantization (from QuIP#/QTIP research) achieve 1.25–1.73 bits/dimension with <5% recall loss: 1M 768-dim vectors shrink from ~3 GB to ~196 MB (H4) / ~124 MB (E8). Note this is the *vector* footprint — the in-RAM HNSW graph adds ~2 KB/vector, so total index RAM at `M=16` is ~2.2 GB/1M (see [Memory Usage at Scale](#memory-usage-at-scale-768-dim-m16)).

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

> **Why the >100M line (and why the default backend doesn't move it):** embedvec's HNSW index and vector cache are held **in RAM**, and the index is **rebuilt on open** by reloading every record from the backend — search never touches disk. Measured resident cost is **~2.2 KB/vector** for H4/E8 and ~5.1 KB for raw at `M=16`, because the graph + metadata dominate the (compressed) vector. So **100M × 768-dim needs ~220 GB even with quantization** (~515 GB raw) — it does *not* fit in 128 GB. Fjall (the default) durably stores 100M+ vectors *on disk* and scales there to terabytes, but that doesn't change the in-RAM index ceiling. Past ~RAM scale (≈ tens of millions on a single node), a disk-paged, horizontally-scalable engine like pgvector wins. See [embedvec + Fjall vs pgvector](#embedvec--fjall-vs-pgvector); migrate via the same `BackendConfig` API.

---

## Why embedvec?

- **Pure Rust** — No C++ dependencies (unless using RocksDB/pgvector), safe and portable
- **Blazing Fast** — AVX2/FMA SIMD acceleration, optimized HNSW with O(1) lookups
- **Memory Efficient** — H4 (~15.7×) and E8 (~24.8×) quantization with <5% recall loss
- **Two Lattice Modes** — E8 (8D, 240 roots) for maximum compression; H4 (4D, 600-cell) for fast decoding
- **Flexible Persistence** — Fjall (default, pure Rust LSM-tree), Sled (pure Rust), RocksDB (high perf), or PostgreSQL/pgvector (distributed)
- **Production Ready** — Async API, metadata filtering, batch operations

---

## Benchmarks

All measurements on 768-dimensional vectors. Run `cargo bench -- lattice` to reproduce.

### Lattice Quantization Comparison (768-dim, 100 vectors per batch)

| Metric | None (raw f32) | H4 (600-cell) | E8 (D8 lattice) |
|--------|---------------|----------------|-----------------|
| **Encode / 100 vectors** | 15.3 µs | 7.26 ms | 3.29 ms |
| **Decode / 100 vectors** | 17.5 µs | 249 µs | 1.10 ms |
| **Insert / 100 vectors** | 21.0 ms | 132 ms (+6.3×) | 501 ms (+24×) |
| **Search / 10 queries (ef=64, 10k DB)** | 2.14 ms | 26.2 ms | 60.3 ms |
| **Bytes / vector (768-dim)** | 3,072 B | **196 B** | **124 B** |
| **Compression ratio** | 1× | **15.7×** | **24.8×** |
| **Bits / dimension** | 32 | ~1.73 | ~1.25 |

> **Quantized search trades speed for memory — for *both* lattices.** The HNSW index stores only node ids and reads vectors from storage during traversal, so every distance computation against a quantized vector **decodes it on the fly**. Raw f32 is fastest (zero-copy, no decode); H4 decodes each candidate (~2.5 µs/vec), and E8 decodes each candidate (~11 µs/vec). Hence the ordering **raw < H4 < E8** for both insert and search. Quantization is a memory/recall optimization, not a latency one — pick it when RAM (not query latency) is the constraint. *(Earlier releases reported H4 search as faster than raw — that was a bug where H4 vectors silently decoded to zeros during graph traversal; fixed in v0.8.)*

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

### Memory Usage at Scale (768-dim, M=16)

embedvec keeps the HNSW graph **and** the vector cache resident in RAM (the index is rebuilt on open). At `M=16` the graph + metadata add **~2 KB/vector**, which *dominates* the compressed vector — quantization shrinks the vectors but not the graph. The columns below are **measured process RSS** (Windows 11, x86-64, small JSON payload per vector; raw confirmed linear at 100k→200k), extrapolated linearly above 200k.

| Mode | Vector bytes | Total RAM/vector | 1M | 10M | 100M |
|------|-------------|------------------|-----|------|------|
| Raw f32 | 3,072 B | ~5.1 KB | ~5.1 GB | ~51 GB | ~515 GB |
| **H4** | **196 B** | **~2.2 KB** | **~2.2 GB** | **~22 GB** | **~220 GB** |
| **E8 10-bit** | **124 B** | **~2.1 KB** | **~2.1 GB** | **~21 GB** | **~213 GB** |

> **What fits in RAM:** ~25M raw, or ~58M H4/E8 vectors per 128 GB. 100M needs ~220 GB even quantized — the graph + metadata, not the vectors, set the ceiling. Building/reopening is single-threaded HNSW insertion (~450 s for 100k H4 here, growing super-linearly), so multi-ten-million indexes already take hours to build *and* to reopen. The per-vector *vector* compression (15.7×/24.8×) is real and lowers the vector share, but total RAM is graph-bound.

### Persistence Backend Comparison (200 B values)

Embedded key-value throughput measured through the `PersistenceBackend` trait — the exact code path embedvec uses for on-disk storage. One-time `open` (~9–15 ms) and clean-shutdown costs are **excluded**; only steady-state work is timed. Reproduce with `cargo bench --bench backend_bench --features persistence-sled` (set `EMBEDVEC_BENCH_N` to change the record count).

**10,000 records:**

| Operation | Fjall (default) | Sled |
|-----------|-----------------|------|
| **Batched bulk-load** (`set_batch`) | **12.0 ms · 836 K/s** | 52.7 ms · 190 K/s |
| Single writes (`set` × N + flush) | 50.4 ms · 198 K/s | 45.7 ms · 219 K/s |
| **Random point `get`** (warm) | **0.78 µs · 1.28 M/s** | 0.98 µs · 1.02 M/s |
| **Prefix scan** (full) | **3.07 ms · 3.26 M/s** | 5.46 ms · 1.83 M/s |

**100,000 records — Fjall's lead widens with scale** (LSM design degrades far more gracefully than a B-tree as data grows):

| Operation | Fjall (default) | Sled | Fjall speedup |
|-----------|-----------------|------|:-------------:|
| **Batched bulk-load** (`set_batch`) | **112 ms · 888 K/s** | 680 ms · 147 K/s | **6.0×** |
| Single writes (`set` × N + flush) | **409 ms · 244 K/s** | 479 ms · 209 K/s | 1.17× |
| **Random point `get`** (warm) | **1.64 µs · 610 K/s** | 1.81 µs · 553 K/s | 1.10× |
| **Prefix scan** (full) | **54 ms · 1.85 M/s** | 85 ms · 1.17 M/s | 1.57× |

> **Why Fjall is the default:** its LSM design wins on reads — point lookups and prefix/range scans (~57–77% faster) — and is **4–6× faster at batched bulk-load** via atomic `set_batch`. Crucially, the lead *grows with scale*: single-key writes flip from ~10% slower than Sled at 10k to ~17% faster at 100k, and batched bulk-load widens from ~4× to ~6×. It is 100% safe Rust with **no C/C++ dependencies**, crash-safe, and actively maintained. embedvec tunes it for vector payloads (32 MiB memtables, compression disabled, configurable block cache via `BackendConfig::cache_size`). Sled remains a solid alternative for small or short-lived stores (lower one-time `open`/shutdown cost); RocksDB (`--features persistence-rocksdb`) needs a C++/libclang toolchain to build. *Measured on Windows 11, x86-64, `bench` profile (LTO).*

---

## Core Features

| Feature | Description |
|---------|-------------|
| **HNSW Indexing** | Hierarchical Navigable Small World graph for O(log n) ANN search |
| **SIMD Distance** | AVX2/FMA accelerated cosine, euclidean, dot product |
| **E8 Quantization** | 8D D8∪D8+½ lattice, 240 roots, ~1.25 bits/dim, 24.8× compression |
| **H4 Quantization** | 4D 600-cell polytope, 120 vertices, ~1.73 bits/dim, 15.7× compression |
| **Metadata Filtering** | Composable filters: eq, gt, lt, contains, AND/OR/NOT |
| **Flexible Persistence** | Fjall (default, pure Rust LSM), Sled (pure Rust), RocksDB (high perf), or pgvector (PostgreSQL) |
| **pgvector Integration** | Native PostgreSQL vector search with HNSW/IVFFlat indexes |
| **Async API** | Tokio-compatible async operations |
| **PyO3 Bindings** | First-class Python support with numpy interop |
| **WASM Support** | Feature-gated for browser/edge deployment |

---

## Quick Start — Rust

```toml
[dependencies]
embedvec = "0.8"   # Fjall persistence backend + async are on by default
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

The Python API also exposes `add(vector, payload) -> id`, `delete(id)` / `delete_many(ids)`, `search_many(query_vectors, ...)` (parallel batch search), `get(id)`, `entries()` (all live `id`/`payload` pairs), and `id in db`. The `filter` argument accepts plain equality (`{"tag": "news"}`) **and** Mongo-style operators: `$eq $ne $gt $gte $lt $lte $in $nin $contains $startswith $endswith $exists`, plus `$and` / `$or` / `$not`, e.g. `{"ts": {"$gte": 1700000000}, "$or": [{"tag": "news"}, {"tag": "blog"}]}`.

### Using embedvec with LangChain / LlamaIndex

embedvec does not ship framework adapters — the Python bindings give you everything needed to write a thin `VectorStore` yourself: `add`/`add_many` (you supply the embeddings), `search`/`search_many` with operator filters, `delete`/`delete_many`, and `entries()`/`get()` to map your framework's string document ids to embedvec's stable integer ids (rebuild the map from `entries()` after reopening a persisted store). Keep document text in the metadata payload.

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
embedvec = { version = "0.8", features = ["persistence-fjall", "async"] }
```

| Feature | Description | Default |
|---------|-------------|---------|
| `persistence-fjall` | On-disk storage via Fjall (pure Rust LSM-tree) | ✓ |
| `persistence-sled` | On-disk storage via Sled (pure Rust) | ✗ |
| `persistence-rocksdb` | On-disk storage via RocksDB (higher perf) | ✗ |
| `persistence-pgvector` | PostgreSQL with native vector search | ✗ |
| `async` | Tokio async API | ✓ |
| `python` | PyO3 bindings | ✗ |
| `simd` | SIMD distance optimizations | ✗ |
| `wasm` | WebAssembly support | ✗ |

---

## Persistence Backends

When a persistence path is configured, every `add` / `add_many` writes the vector (in its compact stored form) plus metadata to the backend — `add_many` does this in a single atomic batch. On open, the store reloads all records and **rebuilds the HNSW index**. It is self-describing: it reopens with the same dimension, distance metric, and quantization it was created with, regardless of constructor arguments. Call `flush()` to force a durable sync.

### Fjall (Default)

Pure Rust log-structured merge-tree (LSM) storage engine — crash-safe, fast on reads and batched writes, with **no C/C++ dependencies**. Enabled by default, so `with_persistence` uses it automatically.

```rust
// Fjall is the default backend — nothing extra to enable
let db = EmbedVec::with_persistence("/path/to/db", 768, Distance::Cosine, 32, 200).await?;
```

```rust
// Explicit form via BackendConfig
let config = BackendConfig::new("/path/to/db").backend(BackendType::Fjall);
let db = EmbedVec::with_backend(config, 768, Distance::Cosine, 32, 200).await?;
```

### Sled (Optional)

```toml
embedvec = { version = "0.8", default-features = false, features = ["persistence-sled", "async"] }
```

```rust
// With Fjall disabled, select the backend explicitly (BackendConfig::new defaults to Fjall)
let config = BackendConfig::new("/path/to/db").backend(BackendType::Sled);
let db = EmbedVec::with_backend(config, 768, Distance::Cosine, 32, 200).await?;
```

### RocksDB (Optional)

Requires a C++/libclang toolchain to build.

```toml
embedvec = { version = "0.8", default-features = false, features = ["persistence-rocksdb", "async"] }
```

```rust
let config = BackendConfig::new("/path/to/db")
    .backend(BackendType::RocksDb)
    .cache_size(256 * 1024 * 1024);
let db = EmbedVec::with_backend(config, 768, Distance::Cosine, 32, 200).await?;
```

### pgvector (PostgreSQL)

```toml
embedvec = { version = "0.8", default-features = false, features = ["persistence-pgvector", "async"] }
```

```rust
let config = BackendConfig::pgvector("postgresql://user:pass@localhost/mydb", 768)
    .table_name("my_vectors")
    .index_type("hnsw");
let backend = PgVectorBackend::connect(&config).await?;
```

---

## embedvec + Fjall vs pgvector

Both can store vectors durably, but they sit at different points on the scale curve. The decisive difference: **embedvec keeps the HNSW index in RAM and rebuilds it on open**, while **pgvector keeps the index on disk and pages it in**.

| Aspect | embedvec + Fjall (default) | pgvector (PostgreSQL) |
|--------|----------------------------|------------------------|
| Deployment | In-process library, zero infra | Client/server database |
| Index location | **RAM** (rebuilt on open) | **Disk**, paged via `shared_buffers` |
| Dataset vs RAM | Must fit in RAM (~2.2 KB/vec at M=16) | Can far exceed RAM |
| Query latency | **Sub-ms** (no network/disk hop) | ~2–10 ms (network + disk + planner) |
| Practical scale (1 node) | ≈ tens of millions (RAM-bound) | **100M+** via partitioning / replicas |
| Startup / build | Re-inserts every vector into RAM (hours at ≳10M) | Index persists on disk; no rebuild |
| Writes | In-process; Fjall batches (~888 K/s bulk) | SQL inserts; MVCC, transactional |
| Concurrency | In-proc readers (`RwLock`) | Many clients, full MVCC |
| Durability / ops | Crash-safe LSM file; you own the file | Mature DB: WAL, backups, replication |
| Best when | Corpus fits in RAM, want sub-ms + no servers | Corpus outgrows RAM, many clients, on-disk index |

**Rule of thumb:** stay on embedvec + Fjall while your corpus comfortably fits in RAM (roughly **≤ tens of millions** of 768-dim vectors per ~128 GB) and you want zero-infra, sub-millisecond search. Move to pgvector when the corpus outgrows RAM, you need many concurrent clients, or you can't afford a multi-hour in-RAM rebuild on every start. The migration path is the same `BackendConfig` API — see [pgvector (PostgreSQL)](#pgvector-postgresql) above.

> Fjall vs pgvector is **not** "embedded vs disk for the *index*." Fjall durably stores the vector *records* on disk (and scales there to terabytes), but embedvec still loads them into an **in-RAM** HNSW graph to serve queries. pgvector is the option when you need the *index itself* to live on disk.

---

## Testing

```bash
cargo test

# Lattice comparison benchmarks only
cargo bench -- lattice

# Persistence backend comparison (Fjall vs Sled)
cargo bench --bench backend_bench --features persistence-sled

# Full benchmark suite
cargo bench
```

---

## Roadmap

- **v0.8** (current): Fjall default backend (pure Rust LSM) with atomic `set_batch`, end-to-end on-disk persistence, H4 search fix, delete support, batch queries (`search_many`), richer metadata-filter operators, and lattice + persistence benchmark suites
- **Future**: Hybrid sparse-dense, full-text + vector, SIMD-accelerated lattice decode, async Python bindings, official LangChain/LlamaIndex adapters

---

## License

MIT — see [LICENSE](LICENSE).

## Contributing

Contributions welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) before submitting PRs.

## Acknowledgments

- HNSW algorithm: Malkov & Yashunin (2016)
- E8 quantization: Inspired by QuIP#, NestQuant, QTIP (2024–2025)
- H4 quantization: Regular 600-cell polytope (icosahedral symmetry in ℝ⁴)
- Rust ecosystem: serde, tokio, pyo3, fjall, sled

---

**embedvec** — The "SQLite of vector search" for Rust-first teams in 2026.
