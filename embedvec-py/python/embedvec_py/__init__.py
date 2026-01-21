"""
embedvec-py: Fast, lightweight vector database with HNSW indexing and E8 quantization

Python bindings for the embedvec Rust crate.

Example:
    >>> import embedvec_py
    >>> import numpy as np
    >>> 
    >>> db = embedvec_py.EmbedVec(dim=768, metric="cosine", m=32, ef_construction=200)
    >>> vectors = np.random.randn(1000, 768).tolist()
    >>> payloads = [{"id": i} for i in range(1000)]
    >>> db.add_many(vectors, payloads)
    >>> 
    >>> query = np.random.randn(768).tolist()
    >>> hits = db.search(query, k=10, ef_search=128)
    >>> for hit in hits:
    ...     print(f"id: {hit['id']}, score: {hit['score']:.4f}")
"""

from .embedvec_py import EmbedVec, __version__

__all__ = ["EmbedVec", "__version__"]
