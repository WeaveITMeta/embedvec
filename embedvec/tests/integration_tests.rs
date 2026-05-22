//! Integration tests for embedvec
//!
//! Tests cover:
//! - Round-trip reconstruction (E8 quantization)
//! - Distance preservation
//! - HNSW search correctness
//! - Filter functionality
//! - Edge cases

use embedvec::{Distance, EmbedVec, FilterExpr, Quantization, E8Codec};
use serde_json::json;

/// Generate random vectors for testing
fn generate_random_vectors(count: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    (0..count)
        .map(|i| {
            (0..dim)
                .map(|j| {
                    let mut hasher = DefaultHasher::new();
                    (seed, i, j).hash(&mut hasher);
                    let h = hasher.finish();
                    // Convert to float in [-1, 1]
                    (h as f32 / u64::MAX as f32) * 2.0 - 1.0
                })
                .collect()
        })
        .collect()
}

/// Normalize a vector to unit length
fn normalize(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-10 {
        v.iter().map(|x| x / norm).collect()
    } else {
        v.to_vec()
    }
}

/// Compute cosine similarity
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a > 1e-10 && norm_b > 1e-10 {
        dot / (norm_a * norm_b)
    } else {
        0.0
    }
}

/// Compute MSE between two vectors
fn mse(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        / a.len() as f32
}

// ============================================================================
// E8 Quantization Tests
// ============================================================================

#[test]
fn test_e8_roundtrip_reconstruction() {
    let codec = E8Codec::new(768, 10, true, 42);
    let vectors = generate_random_vectors(100, 768, 12345);

    let mut total_mse = 0.0;
    for v in &vectors {
        let encoded = codec.encode(v).unwrap();
        let decoded = codec.decode(&encoded);

        let err = mse(v, &decoded);
        total_mse += err;

        // Individual MSE should be reasonable
        assert!(err < 1.0, "MSE too high: {}", err);
    }

    let avg_mse = total_mse / vectors.len() as f32;
    println!("Average MSE: {}", avg_mse);
    assert!(avg_mse < 0.5, "Average MSE too high: {}", avg_mse);
}

#[test]
fn test_e8_roundtrip_normalized_vectors() {
    let codec = E8Codec::new(768, 10, true, 42);
    let vectors: Vec<Vec<f32>> = generate_random_vectors(100, 768, 54321)
        .into_iter()
        .map(|v| normalize(&v))
        .collect();

    let mut total_mse = 0.0;
    for v in &vectors {
        let encoded = codec.encode(v).unwrap();
        let decoded = codec.decode(&encoded);
        total_mse += mse(v, &decoded);
    }

    let avg_mse = total_mse / vectors.len() as f32;
    println!("Average MSE (normalized): {}", avg_mse);
    // Normalized vectors should quantize better
    assert!(avg_mse < 0.3, "Average MSE too high for normalized: {}", avg_mse);
}

#[test]
fn test_e8_distance_preservation() {
    let codec = E8Codec::new(768, 10, true, 42);
    let vectors: Vec<Vec<f32>> = generate_random_vectors(50, 768, 99999)
        .into_iter()
        .map(|v| normalize(&v))
        .collect();

    // Compute original pairwise similarities
    let mut original_sims = Vec::new();
    let mut decoded_sims = Vec::new();

    for i in 0..vectors.len() {
        for j in (i + 1)..vectors.len() {
            let orig_sim = cosine_similarity(&vectors[i], &vectors[j]);

            let dec_i = codec.decode(&codec.encode(&vectors[i]).unwrap());
            let dec_j = codec.decode(&codec.encode(&vectors[j]).unwrap());
            let dec_sim = cosine_similarity(&dec_i, &dec_j);

            original_sims.push(orig_sim);
            decoded_sims.push(dec_sim);
        }
    }

    // Compute Spearman rank correlation (simplified)
    let mut orig_ranked: Vec<(usize, f32)> = original_sims.iter().copied().enumerate().collect();
    let mut dec_ranked: Vec<(usize, f32)> = decoded_sims.iter().copied().enumerate().collect();

    orig_ranked.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    dec_ranked.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    // Check that top-k rankings are mostly preserved
    let top_k = 50;
    let orig_top: std::collections::HashSet<usize> =
        orig_ranked.iter().rev().take(top_k).map(|(i, _)| *i).collect();
    let dec_top: std::collections::HashSet<usize> =
        dec_ranked.iter().rev().take(top_k).map(|(i, _)| *i).collect();

    let overlap = orig_top.intersection(&dec_top).count();
    let overlap_ratio = overlap as f32 / top_k as f32;

    println!("Top-{} overlap ratio: {}", top_k, overlap_ratio);
    assert!(overlap_ratio > 0.7, "Rank preservation too low: {}", overlap_ratio);
}

#[test]
fn test_e8_zero_vector() {
    let codec = E8Codec::new(768, 10, true, 42);
    let zero = vec![0.0f32; 768];

    let encoded = codec.encode(&zero).unwrap();
    let decoded = codec.decode(&encoded);

    // Zero vector should decode to near-zero
    let norm: f32 = decoded.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(norm < 1.0, "Zero vector decoded with high norm: {}", norm);
}

#[test]
fn test_e8_constant_vector() {
    let codec = E8Codec::new(768, 10, true, 42);
    let constant = vec![0.5f32; 768];

    let encoded = codec.encode(&constant).unwrap();
    let decoded = codec.decode(&encoded);

    let err = mse(&constant, &decoded);
    assert!(err < 1.0, "Constant vector MSE too high: {}", err);
}

// ============================================================================
// HNSW Integration Tests
// ============================================================================

#[tokio::test]
async fn test_hnsw_basic_search() {
    let mut db = EmbedVec::new(4, Distance::Euclidean, 16, 100).await.unwrap();

    // Add orthogonal unit vectors
    db.add(&[1.0, 0.0, 0.0, 0.0], json!({"name": "x"})).await.unwrap();
    db.add(&[0.0, 1.0, 0.0, 0.0], json!({"name": "y"})).await.unwrap();
    db.add(&[0.0, 0.0, 1.0, 0.0], json!({"name": "z"})).await.unwrap();
    db.add(&[0.0, 0.0, 0.0, 1.0], json!({"name": "w"})).await.unwrap();

    // Query close to x-axis
    let results = db.search(&[0.9, 0.1, 0.0, 0.0], 2, 50, None).await.unwrap();

    assert!(!results.is_empty());
    assert_eq!(results[0].id, 0); // Should be x-axis vector
}

#[tokio::test]
async fn test_hnsw_recall_vs_bruteforce() {
    let dim = 64;
    let n_vectors = 500;
    let n_queries = 20;
    let k = 10;

    let vectors = generate_random_vectors(n_vectors, dim, 11111);
    let queries = generate_random_vectors(n_queries, dim, 22222);

    // Build HNSW index
    let mut db = EmbedVec::new(dim, Distance::Euclidean, 32, 200).await.unwrap();
    for (i, v) in vectors.iter().enumerate() {
        db.add(v, json!({"id": i})).await.unwrap();
    }

    // Compute brute-force ground truth and HNSW results
    let mut total_recall = 0.0;

    for query in &queries {
        // Brute-force
        let mut distances: Vec<(usize, f32)> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let dist: f32 = query
                    .iter()
                    .zip(v.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f32>()
                    .sqrt();
                (i, dist)
            })
            .collect();
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let bf_top_k: std::collections::HashSet<usize> =
            distances.iter().take(k).map(|(i, _)| *i).collect();

        // HNSW search (high ef for good recall)
        let hnsw_results = db.search(query, k, 200, None).await.unwrap();
        let hnsw_top_k: std::collections::HashSet<usize> =
            hnsw_results.iter().map(|h| h.id).collect();

        let recall = bf_top_k.intersection(&hnsw_top_k).count() as f32 / k as f32;
        total_recall += recall;
    }

    let avg_recall = total_recall / n_queries as f32;
    println!("Average recall@{}: {}", k, avg_recall);
    assert!(avg_recall > 0.9, "Recall too low: {}", avg_recall);
}

#[tokio::test]
async fn test_hnsw_with_e8_quantization() {
    let dim = 128;
    let n_vectors = 200;

    let vectors: Vec<Vec<f32>> = generate_random_vectors(n_vectors, dim, 33333)
        .into_iter()
        .map(|v| normalize(&v))
        .collect();

    // Build with E8 quantization
    let mut db = EmbedVec::builder()
        .dimension(dim)
        .metric(Distance::Cosine)
        .m(16)
        .ef_construction(100)
        .quantization(Quantization::e8(10, true, 42))
        .build()
        .await
        .unwrap();

    for (i, v) in vectors.iter().enumerate() {
        db.add(v, json!({"id": i})).await.unwrap();
    }

    // Search should still work
    let results = db.search(&vectors[0], 5, 100, None).await.unwrap();
    assert!(!results.is_empty());

    // First result should be the query vector itself (or very close)
    assert_eq!(results[0].id, 0);
}

// ============================================================================
// Filter Tests
// ============================================================================

#[tokio::test]
async fn test_filter_exact_match() {
    let mut db = EmbedVec::new(4, Distance::Cosine, 16, 100).await.unwrap();

    db.add(&[1.0, 0.0, 0.0, 0.0], json!({"category": "A"})).await.unwrap();
    db.add(&[0.9, 0.1, 0.0, 0.0], json!({"category": "B"})).await.unwrap();
    db.add(&[0.8, 0.2, 0.0, 0.0], json!({"category": "A"})).await.unwrap();

    let filter = FilterExpr::eq("category", "A");
    let results = db.search(&[1.0, 0.0, 0.0, 0.0], 10, 50, Some(filter)).await.unwrap();

    assert_eq!(results.len(), 2);
    for hit in &results {
        assert_eq!(hit.payload["category"], "A");
    }
}

#[tokio::test]
async fn test_filter_numeric_comparison() {
    let mut db = EmbedVec::new(4, Distance::Cosine, 16, 100).await.unwrap();

    for i in 0..10 {
        db.add(&[1.0, 0.0, 0.0, 0.0], json!({"score": i * 10})).await.unwrap();
    }

    let filter = FilterExpr::gte("score", 50);
    let results = db.search(&[1.0, 0.0, 0.0, 0.0], 10, 50, Some(filter)).await.unwrap();

    assert_eq!(results.len(), 5); // scores 50, 60, 70, 80, 90
    for hit in &results {
        assert!(hit.payload["score"].as_i64().unwrap() >= 50);
    }
}

#[tokio::test]
async fn test_filter_and_or() {
    let mut db = EmbedVec::new(4, Distance::Cosine, 16, 100).await.unwrap();

    db.add(&[1.0, 0.0, 0.0, 0.0], json!({"cat": "A", "val": 10})).await.unwrap();
    db.add(&[0.9, 0.1, 0.0, 0.0], json!({"cat": "B", "val": 20})).await.unwrap();
    db.add(&[0.8, 0.2, 0.0, 0.0], json!({"cat": "A", "val": 30})).await.unwrap();

    // cat == "A" AND val > 15
    let filter = FilterExpr::eq("cat", "A").and(FilterExpr::gt("val", 15));
    let results = db.search(&[1.0, 0.0, 0.0, 0.0], 10, 50, Some(filter)).await.unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].payload["val"], 30);
}

#[tokio::test]
async fn test_filter_in_values() {
    let mut db = EmbedVec::new(4, Distance::Cosine, 16, 100).await.unwrap();

    db.add(&[1.0, 0.0, 0.0, 0.0], json!({"status": "active"})).await.unwrap();
    db.add(&[0.9, 0.1, 0.0, 0.0], json!({"status": "pending"})).await.unwrap();
    db.add(&[0.8, 0.2, 0.0, 0.0], json!({"status": "archived"})).await.unwrap();

    let filter = FilterExpr::in_values("status", vec![json!("active"), json!("pending")]);
    let results = db.search(&[1.0, 0.0, 0.0, 0.0], 10, 50, Some(filter)).await.unwrap();

    assert_eq!(results.len(), 2);
}

// ============================================================================
// Edge Cases
// ============================================================================

#[tokio::test]
async fn test_empty_index_search() {
    let db = EmbedVec::new(4, Distance::Cosine, 16, 100).await.unwrap();
    let results = db.search(&[1.0, 0.0, 0.0, 0.0], 10, 50, None).await.unwrap();
    assert!(results.is_empty());
}

#[tokio::test]
async fn test_dimension_mismatch() {
    let mut db = EmbedVec::new(4, Distance::Cosine, 16, 100).await.unwrap();

    // Wrong dimension should fail
    let result = db.add(&[1.0, 0.0, 0.0], json!({})).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_clear_index() {
    let mut db = EmbedVec::new(4, Distance::Cosine, 16, 100).await.unwrap();

    db.add(&[1.0, 0.0, 0.0, 0.0], json!({})).await.unwrap();
    db.add(&[0.0, 1.0, 0.0, 0.0], json!({})).await.unwrap();

    assert_eq!(db.len().await, 2);

    db.clear().await.unwrap();

    assert_eq!(db.len().await, 0);
    assert!(db.is_empty().await);
}

#[tokio::test]
async fn test_large_batch_add() {
    let dim = 128;
    let n = 1000;

    let mut db = EmbedVec::new(dim, Distance::Cosine, 16, 100).await.unwrap();
    let vectors = generate_random_vectors(n, dim, 44444);
    let payloads: Vec<_> = (0..n).map(|i| json!({"id": i})).collect();

    db.add_many(&vectors, payloads).await.unwrap();

    assert_eq!(db.len().await, n);
}

// ============================================================================
// Memory & Compression Tests
// ============================================================================

#[test]
fn test_e8_memory_reduction() {
    let dim = 768;
    let quant = Quantization::e8(10, true, 42);

    let f32_bytes = dim * 4;
    let e8_bytes = quant.bytes_per_vector(dim);

    let ratio = f32_bytes as f32 / e8_bytes as f32;
    println!("Compression ratio: {}x ({} -> {} bytes)", ratio, f32_bytes, e8_bytes);

    assert!(ratio >= 4.0, "Compression ratio too low: {}", ratio);
}

#[test]
fn test_quantization_bits_per_dim() {
    assert_eq!(Quantization::None.bits_per_dim(), 32.0);
    assert_eq!(Quantization::e8(8, true, 0).bits_per_dim(), 1.0);
    assert_eq!(Quantization::e8(10, true, 0).bits_per_dim(), 1.25);
    assert_eq!(Quantization::e8(12, true, 0).bits_per_dim(), 1.5);
}

// ============================================================================
// End-to-end Persistence Tests
// ============================================================================

/// Vectors + metadata survive a close/reopen cycle, and search still works.
#[cfg(any(feature = "persistence-fjall", feature = "persistence-sled", feature = "persistence-rocksdb"))]
#[tokio::test]
async fn test_persistence_round_trip() {
    let dim = 16;
    let n = 50;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("vectors_db");
    let path_str = path.to_string_lossy().to_string();

    let vectors: Vec<Vec<f32>> = generate_random_vectors(n, dim, 7777)
        .iter()
        .map(|v| normalize(v))
        .collect();
    let payloads: Vec<_> = (0..n).map(|i| json!({"id": i, "tag": "doc"})).collect();

    // Write, flush, and close.
    {
        let mut db = EmbedVec::with_persistence(&path_str, dim, Distance::Cosine, 16, 100)
            .await
            .unwrap();
        db.add_many(&vectors, payloads).await.unwrap();
        db.flush().await.unwrap();
        assert_eq!(db.len().await, n);
    } // db dropped here

    // Reopen the same path: data must be reloaded, not empty.
    let db = EmbedVec::with_persistence(&path_str, dim, Distance::Cosine, 16, 100)
        .await
        .unwrap();
    assert_eq!(db.len().await, n, "reopened database lost vectors");

    // Each stored vector should find itself as the nearest neighbor, with its
    // metadata intact.
    for (expected_id, query) in vectors.iter().enumerate() {
        let hits = db.search(query, 1, 64, None).await.unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, expected_id);
        assert_eq!(hits[0].payload["id"], expected_id);
        assert_eq!(hits[0].payload["tag"], "doc");
    }

    // A metadata filter still applies after reload.
    let filtered = db
        .search(&vectors[0], 5, 64, Some(FilterExpr::eq("tag", "doc")))
        .await
        .unwrap();
    assert!(!filtered.is_empty());
}

/// Deletes survive a close/reopen: deleted ids stay gone, survivors keep their
/// ids + metadata, and deleted ids are not reused for new vectors.
#[cfg(any(feature = "persistence-fjall", feature = "persistence-sled", feature = "persistence-rocksdb"))]
#[tokio::test]
async fn test_delete_persists_across_reopen() {
    let dim = 16;
    let n = 30;
    let dir = tempfile::tempdir().unwrap();
    let path_str = dir.path().join("del_db").to_string_lossy().to_string();

    let vectors: Vec<Vec<f32>> = generate_random_vectors(n, dim, 1234)
        .iter()
        .map(|v| normalize(v))
        .collect();
    let payloads: Vec<_> = (0..n).map(|i| json!({"id": i})).collect();

    // 29 is a trailing delete (exercises the high-water-mark path).
    let to_delete = [3usize, 7, 12, 29];

    {
        let mut db = EmbedVec::with_persistence(&path_str, dim, Distance::Cosine, 16, 100)
            .await
            .unwrap();
        db.add_many(&vectors, payloads).await.unwrap();
        let removed = db.delete_many(&to_delete).await.unwrap();
        assert_eq!(removed, to_delete.len());
        assert_eq!(db.len().await, n - to_delete.len());
        db.flush().await.unwrap();
    }

    // Reopen: live/deleted state and ids must survive.
    let mut db = EmbedVec::with_persistence(&path_str, dim, Distance::Cosine, 16, 100)
        .await
        .unwrap();
    assert_eq!(db.len().await, n - to_delete.len());

    // Deleted vectors are never returned by a search — they are not in the
    // rebuilt graph at all, so this holds regardless of approximate recall.
    for &id in &to_delete {
        let hits = db.search(&vectors[id], 10, 200, None).await.unwrap();
        assert!(
            hits.iter().all(|h| h.id != id),
            "deleted id {id} resurfaced after reopen"
        );
    }

    // Any search returns only live ids, with intact metadata.
    let probe = db.search(&vectors[0], 10, 200, None).await.unwrap();
    assert!(!probe.is_empty());
    for h in &probe {
        assert!(!to_delete.contains(&h.id), "deleted id {} returned", h.id);
        assert_eq!(h.payload["id"], h.id);
    }

    // Per-id live/deleted state persisted exactly: re-deleting a deleted id is a
    // no-op, while re-deleting a survivor succeeds (proving its slot reloaded
    // live). These checks are deterministic, unlike approximate-search recall.
    for &id in &to_delete {
        assert!(!db.delete(id).await.unwrap(), "deleted id {id} came back alive");
    }
    let survivors: Vec<usize> = (0..n).filter(|i| !to_delete.contains(i)).collect();
    for &id in &survivors {
        assert!(db.delete(id).await.unwrap(), "survivor id {id} missing after reopen");
    }
    assert_eq!(db.len().await, 0);

    // A new add never reuses an old id — it gets the high-water mark.
    let new_id = db
        .add(&normalize(&vectors[0]), json!({"id": "new"}))
        .await
        .unwrap();
    assert_eq!(new_id, n, "new id should be the high-water mark, not a reused id");
}

/// A reopened store reports the configuration it was created with, even if the
/// constructor is called with a different quantization.
#[cfg(any(feature = "persistence-fjall", feature = "persistence-sled", feature = "persistence-rocksdb"))]
#[tokio::test]
async fn test_persistence_adopts_stored_config() {
    let dim = 8;
    let dir = tempfile::tempdir().unwrap();
    let path_str = dir.path().join("h4_db").to_string_lossy().to_string();

    {
        let mut db = embedvec::EmbedVec::builder()
            .dimension(dim)
            .metric(Distance::Cosine)
            .quantization(Quantization::h4_default())
            .persistence(&path_str)
            .build()
            .await
            .unwrap();
        db.add(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], json!({"k": "v"}))
            .await
            .unwrap();
        db.flush().await.unwrap();
    }

    // Reopen with the *default* (no) quantization — the H4 config on disk wins.
    let db = embedvec::EmbedVec::builder()
        .dimension(dim)
        .metric(Distance::Cosine)
        .persistence(&path_str)
        .build()
        .await
        .unwrap();
    assert_eq!(db.len().await, 1);
    assert_eq!(*db.quantization(), Quantization::h4_default());
}
