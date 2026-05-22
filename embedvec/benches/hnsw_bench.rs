//! Benchmarks for HNSW index operations
//!
//! ## Table of Contents
//! - **bench_insert**: Insert throughput at 128/384/768 dims (raw only)
//! - **bench_search**: Search at ef=32/64/128/256 (raw index)
//! - **bench_quantization**: E8 encode/decode per 100 vectors
//! - **bench_distance**: Cosine/Euclidean/DotProduct at 768-dim
//! - **bench_lattice_encode**: None vs H4 vs E8 encode throughput (768-dim)
//! - **bench_lattice_decode**: None vs H4 vs E8 decode throughput (768-dim)
//! - **bench_lattice_insert**: None vs H4 vs E8 insert throughput (768-dim, 100 vectors)
//! - **bench_lattice_search**: None vs H4 vs E8 search throughput (768-dim, ef=64, 10k db)
//! - **bench_lattice_memory**: Memory footprint per vector for all three modes
//!
//! Run with: cargo bench
//! Filter to lattice group: cargo bench -- lattice

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use embedvec::{Distance, EmbedVec, Quantization};
use rand::Rng;

fn generate_random_vectors(count: usize, dim: usize) -> Vec<Vec<f32>> {
    let mut rng = rand::thread_rng();
    (0..count)
        .map(|_| (0..dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect())
        .collect()
}

fn bench_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("insert");
    
    for dim in [128, 384, 768].iter() {
        let vectors = generate_random_vectors(1000, *dim);
        
        group.bench_with_input(BenchmarkId::new("raw", dim), dim, |b, _| {
            b.iter(|| {
                let rt = tokio::runtime::Runtime::new().unwrap();
                rt.block_on(async {
                    let mut db = EmbedVec::new(*dim, Distance::Cosine, 16, 100).await.unwrap();
                    for v in &vectors[..100] {
                        db.add(v, serde_json::json!({})).await.unwrap();
                    }
                });
            });
        });
    }
    
    group.finish();
}

fn bench_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("search");
    
    let dim = 768;
    let vectors = generate_random_vectors(10000, dim);
    let queries = generate_random_vectors(100, dim);
    
    // Build index once
    let rt = tokio::runtime::Runtime::new().unwrap();
    let db = rt.block_on(async {
        let mut db = EmbedVec::new(dim, Distance::Cosine, 32, 200).await.unwrap();
        for (i, v) in vectors.iter().enumerate() {
            db.add(v, serde_json::json!({"id": i})).await.unwrap();
        }
        db
    });
    
    for ef_search in [32, 64, 128, 256].iter() {
        group.bench_with_input(BenchmarkId::new("ef", ef_search), ef_search, |b, ef| {
            b.iter(|| {
                rt.block_on(async {
                    for q in &queries[..10] {
                        black_box(db.search(q, 10, *ef, None).await.unwrap());
                    }
                });
            });
        });
    }
    
    group.finish();
}

fn bench_quantization(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantization");
    
    let dim = 768;
    let vectors = generate_random_vectors(1000, dim);
    
    group.bench_function("e8_encode", |b| {
        let codec = embedvec::E8Codec::new(dim, 10, true, 42);
        b.iter(|| {
            for v in &vectors[..100] {
                black_box(codec.encode(v).unwrap());
            }
        });
    });
    
    group.bench_function("e8_decode", |b| {
        let codec = embedvec::E8Codec::new(dim, 10, true, 42);
        let encoded: Vec<_> = vectors[..100].iter().map(|v| codec.encode(v).unwrap()).collect();
        b.iter(|| {
            for e in &encoded {
                black_box(codec.decode(e));
            }
        });
    });
    
    group.finish();
}

// ---------------------------------------------------------------------------
// Lattice comparison benchmarks: None vs H4 vs E8 side-by-side
// ---------------------------------------------------------------------------

/// Encode 100 vectors: raw copy (None), H4 encode, E8 encode
fn bench_lattice_encode(c: &mut Criterion) {
    let mut group = c.benchmark_group("lattice_encode");
    let dim = 768;
    let vectors = generate_random_vectors(1000, dim);

    // None — baseline: raw Vec<f32> clone (no codec work)
    group.bench_function("none", |b| {
        b.iter(|| {
            for v in &vectors[..100] {
                black_box(v.clone());
            }
        });
    });

    // H4 — 4D 600-cell quantization
    group.bench_function("h4", |b| {
        let codec = embedvec::H4Codec::new(dim, true, 0xdeadbeef);
        b.iter(|| {
            for v in &vectors[..100] {
                black_box(codec.encode(v).unwrap());
            }
        });
    });

    // E8 — 8D D8∪D8+½ lattice quantization
    group.bench_function("e8", |b| {
        let codec = embedvec::E8Codec::new(dim, 10, true, 0xcafef00d);
        b.iter(|| {
            for v in &vectors[..100] {
                black_box(codec.encode(v).unwrap());
            }
        });
    });

    group.finish();
}

/// Decode 100 vectors: raw clone (None), H4 decode, E8 decode
fn bench_lattice_decode(c: &mut Criterion) {
    let mut group = c.benchmark_group("lattice_decode");
    let dim = 768;
    let vectors = generate_random_vectors(1000, dim);

    // None — baseline: raw Vec<f32> clone
    group.bench_function("none", |b| {
        b.iter(|| {
            for v in &vectors[..100] {
                black_box(v.clone());
            }
        });
    });

    // H4
    group.bench_function("h4", |b| {
        let codec = embedvec::H4Codec::new(dim, true, 0xdeadbeef);
        let encoded: Vec<_> = vectors[..100].iter().map(|v| codec.encode(v).unwrap()).collect();
        b.iter(|| {
            for e in &encoded {
                black_box(codec.decode(e));
            }
        });
    });

    // E8
    group.bench_function("e8", |b| {
        let codec = embedvec::E8Codec::new(dim, 10, true, 0xcafef00d);
        let encoded: Vec<_> = vectors[..100].iter().map(|v| codec.encode(v).unwrap()).collect();
        b.iter(|| {
            for e in &encoded {
                black_box(codec.decode(e));
            }
        });
    });

    group.finish();
}

/// Insert 100 vectors into a fresh EmbedVec: None vs H4 vs E8
fn bench_lattice_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("lattice_insert");
    // Reduce sample size — each iteration builds a new DB
    group.sample_size(20);
    let dim = 768;
    let vectors = generate_random_vectors(200, dim);
    let rt = tokio::runtime::Runtime::new().unwrap();

    for (label, quant) in [
        ("none", Quantization::None),
        ("h4",   Quantization::h4_default()),
        ("e8",   Quantization::e8_default()),
    ] {
        let quant_clone = quant.clone();
        group.bench_function(label, |b| {
            b.iter(|| {
                rt.block_on(async {
                    let mut db = EmbedVec::builder()
                        .dimension(dim)
                        .metric(Distance::Cosine)
                        .quantization(quant_clone.clone())
                        .build()
                        .await
                        .unwrap();
                    for v in &vectors[..100] {
                        db.add(black_box(v), serde_json::json!({})).await.unwrap();
                    }
                    black_box(db);
                });
            });
        });
    }

    group.finish();
}

/// Search ef=64, k=10 across a 10k-vector DB: None vs H4 vs E8
fn bench_lattice_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("lattice_search");
    let dim = 768;
    let db_vectors = generate_random_vectors(10_000, dim);
    let queries = generate_random_vectors(50, dim);
    let rt = tokio::runtime::Runtime::new().unwrap();

    for (label, quant) in [
        ("none", Quantization::None),
        ("h4",   Quantization::h4_default()),
        ("e8",   Quantization::e8_default()),
    ] {
        // Build the DB once per lattice mode
        let db = rt.block_on(async {
            let mut db = EmbedVec::builder()
                .dimension(dim)
                .metric(Distance::Cosine)
                .quantization(quant.clone())
                .build()
                .await
                .unwrap();
            for (i, v) in db_vectors.iter().enumerate() {
                db.add(v, serde_json::json!({"i": i})).await.unwrap();
            }
            db
        });

        group.bench_function(label, |b| {
            b.iter(|| {
                rt.block_on(async {
                    for q in &queries[..10] {
                        black_box(db.search(q, 10, 64, None).await.unwrap());
                    }
                });
            });
        });
    }

    group.finish();
}

/// Memory footprint: bytes per stored vector for None/H4/E8 at 768-dim
///
/// Not a timing benchmark — uses bench_function to emit a single consistent
/// measurement via black_box so Criterion records it as a data point.
fn bench_lattice_memory(c: &mut Criterion) {
    let mut group = c.benchmark_group("lattice_memory_bytes_per_vector");
    let dim = 768;
    let vectors = generate_random_vectors(100, dim);
    let rt = tokio::runtime::Runtime::new().unwrap();

    for (label, quant) in [
        ("none", Quantization::None),
        ("h4",   Quantization::h4_default()),
        ("e8",   Quantization::e8_default()),
    ] {
        group.bench_function(label, |b| {
            b.iter(|| {
                rt.block_on(async {
                    let mut db = EmbedVec::builder()
                        .dimension(dim)
                        .metric(Distance::Cosine)
                        .quantization(quant.clone())
                        .build()
                        .await
                        .unwrap();
                    for v in &vectors {
                        db.add(v, serde_json::json!({})).await.unwrap();
                    }
                    let storage = db.storage.read();
                    black_box(storage.memory_bytes());
                });
            });
        });
    }

    group.finish();
}

fn bench_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("distance");
    
    let dim = 768;
    let vectors = generate_random_vectors(1000, dim);
    
    group.bench_function("cosine_768d", |b| {
        b.iter(|| {
            for i in 0..100 {
                black_box(embedvec::Distance::Cosine.compute(&vectors[i], &vectors[i + 100]));
            }
        });
    });
    
    group.bench_function("euclidean_768d", |b| {
        b.iter(|| {
            for i in 0..100 {
                black_box(embedvec::Distance::Euclidean.compute(&vectors[i], &vectors[i + 100]));
            }
        });
    });
    
    group.bench_function("dot_product_768d", |b| {
        b.iter(|| {
            for i in 0..100 {
                black_box(embedvec::Distance::DotProduct.compute(&vectors[i], &vectors[i + 100]));
            }
        });
    });
    
    group.finish();
}

criterion_group!(
    benches,
    bench_insert,
    bench_search,
    bench_quantization,
    bench_distance,
    bench_lattice_encode,
    bench_lattice_decode,
    bench_lattice_insert,
    bench_lattice_search,
    bench_lattice_memory,
);
criterion_main!(benches);
