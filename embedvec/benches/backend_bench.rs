//! Persistence backend benchmarks: Fjall (default) vs Sled vs RocksDB
//!
//! ## Table of Contents
//! - **backend_insert_single**: Steady-state per-key write throughput (10k `set` + flush)
//! - **backend_insert_batch**: Atomic batched bulk-load throughput (`set_batch` + flush)
//! - **backend_get**: Random point-lookup latency on a warm 10k-pair store
//! - **backend_scan_prefix**: Full prefix-scan throughput over 10k pairs
//!
//! Each backend is exercised through the `PersistenceBackend` trait, so the
//! numbers reflect the exact code path embedvec uses for on-disk storage.
//! Values are 200 bytes — roughly an H4-quantized 768-dim vector record.
//!
//! The write benchmarks open the store in (untimed) setup and return the handle
//! from the (timed) routine, so the one-time `open` and clean-`Drop`/shutdown
//! costs are excluded — only steady-state write + durable flush is measured.
//!
//! Run (default = Fjall only):
//!     cargo bench --bench backend_bench
//! Compare Fjall vs Sled (both pure Rust):
//!     cargo bench --bench backend_bench --features persistence-sled
//! Add RocksDB (needs clang/libclang to build):
//!     cargo bench --bench backend_bench --features "persistence-sled persistence-rocksdb"

use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion, Throughput};
use embedvec::persistence::{create_backend, BackendConfig, BackendType, PersistenceBackend};
use rand::Rng;
use tempfile::TempDir;

/// Payload size in bytes — approximates an H4-quantized 768-dim vector record.
const VALUE_LEN: usize = 200;

/// Number of key-value pairs per benchmark.
///
/// Defaults to 10k; override for larger-scale runs, e.g.
/// `EMBEDVEC_BENCH_N=100000 cargo bench --bench backend_bench --features persistence-sled`.
fn bench_n() -> usize {
    std::env::var("EMBEDVEC_BENCH_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10_000)
}

/// All persistence backends compiled into this build.
fn enabled_backends() -> Vec<(&'static str, BackendType)> {
    #[allow(unused_mut)]
    let mut backends: Vec<(&'static str, BackendType)> = Vec::new();
    #[cfg(feature = "persistence-fjall")]
    backends.push(("fjall", BackendType::Fjall));
    #[cfg(feature = "persistence-sled")]
    backends.push(("sled", BackendType::Sled));
    #[cfg(feature = "persistence-rocksdb")]
    backends.push(("rocksdb", BackendType::RocksDb));
    backends
}

fn make_pairs(n: usize) -> Vec<(Vec<u8>, Vec<u8>)> {
    let mut rng = rand::thread_rng();
    (0..n)
        .map(|i| {
            let key = format!("vec:{:08}", i).into_bytes();
            let value: Vec<u8> = (0..VALUE_LEN).map(|_| rng.gen::<u8>()).collect();
            (key, value)
        })
        .collect()
}

fn open_backend(backend_type: BackendType, dir: &TempDir) -> Box<dyn PersistenceBackend> {
    let config =
        BackendConfig::new(dir.path().to_string_lossy().to_string()).backend(backend_type);
    create_backend(&config).expect("failed to open backend")
}

/// Open a warm store pre-populated with all pairs (returns dir to keep it alive).
fn warm_store(
    backend_type: BackendType,
    pairs: &[(Vec<u8>, Vec<u8>)],
) -> (TempDir, Box<dyn PersistenceBackend>) {
    let dir = TempDir::new().unwrap();
    let backend = open_backend(backend_type, &dir);
    backend.set_batch(pairs).unwrap();
    backend.flush().unwrap();
    (dir, backend)
}

/// Steady-state per-key writes: `set` x N then one durable flush.
/// Open and Drop happen in untimed setup/teardown.
fn bench_insert_single(c: &mut Criterion) {
    let n = bench_n();
    let pairs = make_pairs(n);
    let mut group = c.benchmark_group("backend_insert_single");
    group.sample_size(10);
    group.throughput(Throughput::Elements(n as u64));

    for (label, backend_type) in enabled_backends() {
        group.bench_function(label, |b| {
            b.iter_batched(
                || {
                    let dir = TempDir::new().unwrap();
                    let backend = open_backend(backend_type, &dir);
                    (dir, backend)
                },
                |(dir, backend)| {
                    for (k, v) in &pairs {
                        backend.set(k, v).unwrap();
                    }
                    backend.flush().unwrap();
                    (dir, backend) // returned -> Drop runs outside the timed section
                },
                BatchSize::PerIteration,
            );
        });
    }
    group.finish();
}

/// Atomic batched bulk-load: a single `set_batch` then one durable flush.
fn bench_insert_batch(c: &mut Criterion) {
    let n = bench_n();
    let pairs = make_pairs(n);
    let mut group = c.benchmark_group("backend_insert_batch");
    group.sample_size(10);
    group.throughput(Throughput::Elements(n as u64));

    for (label, backend_type) in enabled_backends() {
        group.bench_function(label, |b| {
            b.iter_batched(
                || {
                    let dir = TempDir::new().unwrap();
                    let backend = open_backend(backend_type, &dir);
                    (dir, backend)
                },
                |(dir, backend)| {
                    backend.set_batch(&pairs).unwrap();
                    backend.flush().unwrap();
                    (dir, backend)
                },
                BatchSize::PerIteration,
            );
        });
    }
    group.finish();
}

/// Random point lookups against a warm store.
fn bench_get(c: &mut Criterion) {
    let n = bench_n();
    let pairs = make_pairs(n);
    let mut group = c.benchmark_group("backend_get");

    for (label, backend_type) in enabled_backends() {
        let (_dir, backend) = warm_store(backend_type, &pairs);
        let mut rng = rand::thread_rng();
        group.bench_function(label, |b| {
            b.iter(|| {
                let idx = rng.gen_range(0..n);
                black_box(backend.get(&pairs[idx].0).unwrap());
            });
        });
    }
    group.finish();
}

/// Full prefix scan returning all 10k pairs.
fn bench_scan_prefix(c: &mut Criterion) {
    let n = bench_n();
    let pairs = make_pairs(n);
    let mut group = c.benchmark_group("backend_scan_prefix");
    group.sample_size(20);
    group.throughput(Throughput::Elements(n as u64));

    for (label, backend_type) in enabled_backends() {
        let (_dir, backend) = warm_store(backend_type, &pairs);
        group.bench_function(label, |b| {
            b.iter(|| {
                black_box(backend.scan_prefix(b"vec:").unwrap().len());
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_insert_single,
    bench_insert_batch,
    bench_get,
    bench_scan_prefix
);
criterion_main!(benches);
