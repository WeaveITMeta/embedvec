# embedvec-py

Python bindings for [embedvec](https://github.com/embedvec/embedvec) — a fast, lightweight, in-process vector database with HNSW indexing and E8 lattice quantization.

## Installation

```bash
pip install embedvec-py
```

## Quick Start

```python
import embedvec_py
import numpy as np

# Create database
db = embedvec_py.EmbedVec(
    dim=768,
    metric="cosine",           # "cosine", "euclidean", or "dot"
    m=32,                      # HNSW connections per layer
    ef_construction=200,       # HNSW build parameter
    quantization="e8p-10bit",  # Optional: "e8p-8bit", "e8p-10bit", "e8p-12bit"
    persist_path=None,         # Optional: path for persistence
)

# Add vectors
vectors = np.random.randn(10000, 768).tolist()
payloads = [{"doc_id": str(i), "category": "news" if i % 2 == 0 else "blog"} 
            for i in range(10000)]

db.add_many(vectors, payloads)

# Search
query = np.random.randn(768).tolist()
hits = db.search(
    query_vector=query,
    k=10,
    ef_search=128,
    filter={"category": "news"}  # Optional filter
)

for hit in hits:
    print(f"Score: {hit['score']:.4f} | ID: {hit['id']} | {hit['payload']}")
```

## Features

- **Fast**: Pure Rust core with HNSW indexing
- **Memory Efficient**: E8 quantization provides 4-6× memory reduction
- **Flexible**: Metadata filtering with exact match, range, and boolean expressions
- **Persistent**: Optional on-disk storage
- **Easy**: Simple Python API with numpy interop

## API Reference

### EmbedVec

```python
db = embedvec_py.EmbedVec(
    dim: int,                    # Vector dimension
    metric: str = "cosine",      # Distance metric
    m: int = 32,                 # HNSW M parameter
    ef_construction: int = 200,  # HNSW build parameter
    quantization: str = None,    # Quantization mode
    persist_path: str = None,    # Persistence path
)
```

### Methods

- `add(vector, payload)` - Add single vector
- `add_many(vectors, payloads)` - Batch add
- `search(query_vector, k, ef_search, filter)` - Find k nearest neighbors
- `len()` - Number of vectors
- `clear()` - Remove all vectors
- `memory_bytes()` - Memory usage
- `compression_ratio()` - Compression ratio vs FP32

## License

MIT
