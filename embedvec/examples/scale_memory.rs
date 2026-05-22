// Measure the *resident* memory footprint of an in-memory embedvec index.
//
// embedvec keeps the HNSW graph + vector cache in RAM, so this reports the real
// per-vector cost (vectors + graph + metadata + allocator overhead), not just
// the on-disk vector bytes.
//
//   cargo run --release --example scale_memory -- <raw|h4|e8> <n> [dim]
//
// Example: cargo run --release --example scale_memory -- h4 1000000

use embedvec::{Distance, EmbedVec, Quantization};
use std::time::Instant;

fn rss_bytes() -> u64 {
    memory_stats::memory_stats()
        .map(|m| m.physical_mem as u64)
        .unwrap_or(0)
}

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(|s| s.as_str()).unwrap_or("h4");
    let n: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(100_000);
    let dim: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(768);

    let quant = match mode {
        "raw" | "none" => Quantization::None,
        "h4" => Quantization::h4_default(),
        "e8" => Quantization::e8_default(),
        other => {
            eprintln!("unknown mode {other}; use raw|h4|e8");
            return;
        }
    };

    let base = rss_bytes();

    let mut db = EmbedVec::builder()
        .dimension(dim)
        .metric(Distance::Cosine)
        .m(16)
        .ef_construction(100)
        .quantization(quant)
        .build()
        .await
        .unwrap();

    // Deterministic pseudo-random vectors, generated on the fly to avoid holding
    // a second copy of the whole dataset in RAM while we measure.
    let mut state: u64 = 0x9E3779B97F4A7C15;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state as f32 / u64::MAX as f32) * 2.0 - 1.0
    };

    let t = Instant::now();
    let mut v = vec![0.0f32; dim];
    for i in 0..n {
        for x in v.iter_mut() {
            *x = next();
        }
        db.add_internal(&v, serde_json::json!({ "i": i })).unwrap();
    }
    let build = t.elapsed();

    let peak = rss_bytes();
    let used = peak.saturating_sub(base);
    let vec_bytes = db.storage.read().memory_bytes();

    println!(
        "mode={mode:>4} n={n:>9} dim={dim}  build={build:>8.1?}  \
         RSS_total={:.2} GB  RSS_used={:.2} GB  bytes/vec(RSS)={:.0}  vec_bytes/vec={:.0}",
        peak as f64 / 1e9,
        used as f64 / 1e9,
        used as f64 / n as f64,
        vec_bytes as f64 / n as f64,
    );
}
