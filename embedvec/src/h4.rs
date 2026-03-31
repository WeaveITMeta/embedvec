//! H4 Lattice Quantization Module
//!
//! ## Table of Contents
//! - **H4Codec**: Main codec for encoding/decoding vectors using the H4 polytope lattice
//! - **H4Oracle**: Nearest 600-cell vertex finder (120 minimal vectors in 4D)
//! - **H4Point**: Compact representation of a 600-cell vertex (4 dimensions)
//! - **HadamardTransform4**: Fast 4D Walsh-Hadamard transform for preprocessing
//! - **generate_h4_vertices**: Enumerates all 120 vertices of the 600-cell
//!
//! ## Background
//! The H4 root system lives in ℝ⁴ and has 120 minimal vectors (roots), which are
//! the vertices of the regular 600-cell polytope. This is the densest known lattice
//! quantizer in 4D and exploits icosahedral / icosian symmetry.
//!
//! The 120 vertices of the 600-cell are:
//! - 24 permutations of (±1, 0, 0, 0)                          [Type A: 24 vectors]
//! - 96 permutations of (±½, ±½, ±½, ±½)                       [Type B: 16 vectors]
//! - 80 permutations of (0, ±½, ±1/2φ, ±φ/2) where φ = golden ratio  [Type C: 96 vectors]
//!
//! Wait — correct enumeration:
//! - 8 permutations of (±1, 0, 0, 0)
//! - 16 all-sign variants of (½, ½, ½, ½)
//! - 96 even permutations of (0, ½, 1/(2φ), φ/2) with all sign combos
//!
//! The correct 120 vertices of the 600-cell are:
//! - 8 × (±1, 0, 0, 0) and all coordinate permutations → 8 total
//! - 16 × (±½, ±½, ±½, ±½) → 16 total
//! - 96 × even permutations of (0, ±½, ±1/(2φ), ±φ/2) → 96 total
//! Total: 8 + 16 + 96 = 120 ✓
//!
//! ## Algorithm Overview
//! 1. Apply 4D Hadamard transform + random signs to each 4D block
//! 2. Split high-dim vector into 4D blocks
//! 3. Quantize each block to nearest 600-cell vertex via H4Oracle
//! 4. Store compact 7-bit index (120 vertices → fits in u8)
//! 5. Decode on-the-fly during distance computation (asymmetric search)
//!
//! ## Comparison with E8
//! | Property          | E8              | H4               |
//! |-------------------|-----------------|------------------|
//! | Dimension         | 8D blocks        | 4D blocks        |
//! | Minimal vectors   | 240             | 120              |
//! | Bits per block    | ~10 bits         | ~7 bits          |
//! | Kissing number    | 240             | 120              |
//! | Best for dims     | multiples of 8   | multiples of 4   |

use crate::error::{EmbedVecError, Result};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

/// 1 / (2φ) = (√5 - 1) / 4, where φ = (1 + √5) / 2 is the golden ratio
const INV_2PHI: f32 = 0.309_017_0;

/// φ / 2
const HALF_PHI: f32 = 0.809_016_9;

/// H4 600-cell has 120 minimal vectors (vertices)
pub const H4_NUM_VERTICES: usize = 120;

/// Block size for H4 quantization (4 dimensions)
pub const H4_BLOCK_SIZE: usize = 4;

/// H4 Codec for encoding and decoding vectors
///
/// Handles the full pipeline: 4D Hadamard preprocessing, block-wise H4 quantization
/// to 600-cell vertices, and compact 8-bit index storage.
#[derive(Debug, Clone)]
pub struct H4Codec {
    /// Vector dimension
    dimension: usize,
    /// Number of 4D blocks
    num_blocks: usize,
    /// Whether to use Hadamard preprocessing
    use_hadamard: bool,
    /// Random signs for Hadamard preprocessing (one per padded dimension)
    random_signs: Vec<f32>,
    /// Precomputed 600-cell vertices (120 × 4D)
    vertices: Vec<[f32; 4]>,
}

impl H4Codec {
    /// Create a new H4 codec
    ///
    /// # Arguments
    /// * `dimension` - Vector dimension (will be padded to multiple of 4)
    /// * `use_hadamard` - Apply 4D Hadamard + random signs preprocessing
    /// * `random_seed` - Seed for reproducible random signs
    pub fn new(dimension: usize, use_hadamard: bool, random_seed: u64) -> Self {
        let num_blocks = (dimension + H4_BLOCK_SIZE - 1) / H4_BLOCK_SIZE;
        let padded_dim = num_blocks * H4_BLOCK_SIZE;

        // Generate random signs for preprocessing
        let mut rng = ChaCha8Rng::seed_from_u64(random_seed);
        let random_signs: Vec<f32> = (0..padded_dim)
            .map(|_| if rand::Rng::gen::<bool>(&mut rng) { 1.0 } else { -1.0 })
            .collect();

        // Precompute 600-cell vertices once at construction time
        let vertices = generate_h4_vertices();

        Self {
            dimension,
            num_blocks,
            use_hadamard,
            random_signs,
            vertices,
        }
    }

    /// Encode a vector to H4 quantized codes
    ///
    /// # Arguments
    /// * `vector` - Input vector (f32)
    ///
    /// # Returns
    /// H4-quantized encoded vector with per-block vertex indices and scale
    pub fn encode(&self, vector: &[f32]) -> Result<H4EncodedVector> {
        if vector.len() != self.dimension {
            return Err(EmbedVecError::DimensionMismatch {
                expected: self.dimension,
                got: vector.len(),
            });
        }

        // Pad vector to multiple of 4
        let mut padded = vec![0.0f32; self.num_blocks * H4_BLOCK_SIZE];
        padded[..vector.len()].copy_from_slice(vector);

        // Apply random signs
        for (i, v) in padded.iter_mut().enumerate() {
            *v *= self.random_signs[i];
        }

        // Apply 4D Hadamard transform to each block
        if self.use_hadamard {
            for block in padded.chunks_mut(H4_BLOCK_SIZE) {
                hadamard4_inplace(block);
            }
        }

        // Compute global scale factor — normalize so max coordinate ≈ 1.0 (600-cell radius)
        let max_abs = padded.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        let scale = if max_abs > 1e-10 { max_abs } else { 1.0 };

        // Quantize each 4D block to nearest 600-cell vertex
        let mut indices = Vec::with_capacity(self.num_blocks);
        for block in padded.chunks(H4_BLOCK_SIZE) {
            let normalized: [f32; 4] = std::array::from_fn(|i| block[i] / scale);
            let idx = H4Oracle::nearest_vertex_index(&normalized, &self.vertices);
            indices.push(idx);
        }

        Ok(H4EncodedVector { indices, scale })
    }

    /// Decode H4 codes back to approximate vector
    ///
    /// # Arguments
    /// * `encoded` - Encoded vector with vertex indices and scale
    ///
    /// # Returns
    /// Reconstructed f32 vector
    pub fn decode(&self, encoded: &H4EncodedVector) -> Vec<f32> {
        let mut result = Vec::with_capacity(self.num_blocks * H4_BLOCK_SIZE);

        // Decode each 4D block from vertex index
        for &idx in &encoded.indices {
            let vertex = &self.vertices[idx as usize];
            for &v in vertex {
                result.push(v * encoded.scale);
            }
        }

        // Apply inverse 4D Hadamard transform
        if self.use_hadamard {
            for block in result.chunks_mut(H4_BLOCK_SIZE) {
                hadamard4_inplace(block);
            }
        }

        // Apply inverse random signs
        for (i, v) in result.iter_mut().enumerate() {
            *v *= self.random_signs[i];
        }

        // Truncate to original dimension
        result.truncate(self.dimension);
        result
    }

    /// Compute asymmetric distance between f32 query and encoded H4 vector
    ///
    /// Query remains in f32; database vector is decoded on-the-fly for accuracy.
    pub fn asymmetric_distance(&self, query: &[f32], encoded: &H4EncodedVector) -> f32 {
        let decoded = self.decode(encoded);
        let mut dist = 0.0f32;
        for (q, d) in query.iter().zip(decoded.iter()) {
            let diff = q - d;
            dist += diff * diff;
        }
        dist.sqrt()
    }

    /// Get memory usage per encoded vector in bytes
    ///
    /// Each block uses 1 byte (u8 index into 120 vertices) + 4 bytes for scale.
    pub fn bytes_per_vector(&self) -> usize {
        self.num_blocks + 4 // num_blocks × u8 + f32 scale
    }

    /// Get number of 4D blocks
    pub fn num_blocks(&self) -> usize {
        self.num_blocks
    }

    /// Get bits per dimension (log2(120) / 4 ≈ 1.72 bits/dim)
    pub fn bits_per_dim(&self) -> f32 {
        (H4_NUM_VERTICES as f32).log2() / H4_BLOCK_SIZE as f32
    }
}

/// Encoded H4 vector representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct H4EncodedVector {
    /// Index into the 600-cell vertex table for each 4D block (0..119)
    pub indices: Vec<u8>,
    /// Global scale factor for reconstruction
    pub scale: f32,
}

impl H4EncodedVector {
    /// Create empty encoded vector
    pub fn empty() -> Self {
        Self {
            indices: Vec::new(),
            scale: 1.0,
        }
    }

    /// Get serialized size in bytes
    pub fn size_bytes(&self) -> usize {
        // indices: 1 byte each, plus 4 bytes for scale (f32)
        self.indices.len() + 4
    }
}

/// H4 Lattice Oracle
///
/// Finds the nearest 600-cell vertex to a given normalized 4D vector.
/// The 600-cell has 120 vertices, all at unit distance from origin after normalization.
pub struct H4Oracle;

impl H4Oracle {
    /// Find the index of the nearest 600-cell vertex to the input 4D vector
    ///
    /// Uses exhaustive search over 120 vertices — fast since it's a small fixed set.
    ///
    /// # Arguments
    /// * `x` - 4D input vector (should be approximately unit-normalized)
    /// * `vertices` - Precomputed vertex table (120 × 4D)
    ///
    /// # Returns
    /// Index into the vertex table (0..119)
    #[inline]
    pub fn nearest_vertex_index(x: &[f32; 4], vertices: &[[f32; 4]]) -> u8 {
        let mut best_idx = 0usize;
        let mut best_dist = f32::MAX;

        for (idx, vertex) in vertices.iter().enumerate() {
            let dist = Self::squared_distance_4d(x, vertex);
            if dist < best_dist {
                best_dist = dist;
                best_idx = idx;
            }
        }

        best_idx as u8
    }

    /// Find the nearest vertex and return the vertex coordinates directly
    ///
    /// Convenience wrapper returning coordinates instead of index.
    #[inline]
    pub fn nearest_vertex(x: &[f32; 4], vertices: &[[f32; 4]]) -> [f32; 4] {
        let idx = Self::nearest_vertex_index(x, vertices);
        vertices[idx as usize]
    }

    /// Squared Euclidean distance between two 4D vectors — unrolled for speed
    #[inline(always)]
    fn squared_distance_4d(a: &[f32; 4], b: &[f32; 4]) -> f32 {
        let d0 = a[0] - b[0];
        let d1 = a[1] - b[1];
        let d2 = a[2] - b[2];
        let d3 = a[3] - b[3];
        d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3
    }
}

/// Fast in-place 4D Walsh-Hadamard Transform
///
/// Normalized so that two applications return the original vector (involution).
/// Transform matrix: H₄ = (1/2) × [[1,1,1,1],[1,-1,1,-1],[1,1,-1,-1],[1,-1,-1,1]]
/// Applied as butterfly network (log₂4 = 2 stages).
#[inline]
pub fn hadamard4_inplace(data: &mut [f32]) {
    debug_assert_eq!(data.len(), 4);

    // Stage 1: size-2 butterflies
    let (a, b, c, d) = (data[0], data[1], data[2], data[3]);
    let (s0, s1) = (a + b, a - b);
    let (s2, s3) = (c + d, c - d);

    // Stage 2: size-4 butterflies
    data[0] = (s0 + s2) * 0.5;
    data[1] = (s1 + s3) * 0.5;
    data[2] = (s0 - s2) * 0.5;
    data[3] = (s1 - s3) * 0.5;
}

/// Generate all 120 vertices of the regular 600-cell in ℝ⁴
///
/// The 600-cell is the 4D analogue of the icosahedron. Its 120 vertices consist of:
///
/// **Type A** — 8 axis-aligned vertices at distance 1:
///   (±1, 0, 0, 0) and all coordinate permutations → 8 total
///
/// **Type B** — 16 vertices at distance 1:
///   (±½, ±½, ±½, ±½) with all 16 sign combinations → 16 total
///
/// **Type C** — 96 icosahedral vertices at distance 1:
///   Even permutations of (0, ±½, ±INV_2PHI, ±HALF_PHI) → 96 total
///   where φ = (1+√5)/2 is the golden ratio
///
/// All 120 vertices lie on the unit 3-sphere (‖v‖ = 1).
pub fn generate_h4_vertices() -> Vec<[f32; 4]> {
    let mut vertices = Vec::with_capacity(H4_NUM_VERTICES);

    // --- Type A: (±1, 0, 0, 0) permutations — 8 vectors ---
    for pos in 0..4 {
        for sign in [-1.0f32, 1.0] {
            let mut v = [0.0f32; 4];
            v[pos] = sign;
            vertices.push(v);
        }
    }
    // Count after Type A: 8

    // --- Type B: (±½, ±½, ±½, ±½) all sign combinations — 16 vectors ---
    for mask in 0u8..16 {
        let v: [f32; 4] = std::array::from_fn(|i| {
            if (mask >> i) & 1 == 1 { -0.5 } else { 0.5 }
        });
        vertices.push(v);
    }
    // Count after Type B: 24

    // --- Type C: even permutations of (0, ±½, ±INV_2PHI, ±HALF_PHI) — 96 vectors ---
    // The 3 non-zero values with all sign combinations: 2³ = 8 sign combos × 12 even perms = 96
    // Even permutations of [0, 1, 2, 3] starting from (0, a, b, c):
    //   Position assignments where 0 is at index i, then abc fill the remaining 3 in sorted order.
    //   The 12 even permutations of 4 elements where element "0" appears at each position:
    //   (0,a,b,c), (b,0,a,c) [wrong — need actual even perms]
    // Correct approach: enumerate all 24 permutations, keep only the 12 even ones.
    let base = [0.0f32, 0.5, INV_2PHI, HALF_PHI];
    let even_perms: [[usize; 4]; 12] = [
        [0, 1, 2, 3],
        [0, 2, 3, 1],
        [0, 3, 1, 2],
        [1, 0, 3, 2],
        [1, 2, 0, 3],
        [1, 3, 2, 0],
        [2, 0, 1, 3],
        [2, 1, 3, 0],
        [2, 3, 0, 1],
        [3, 0, 2, 1],
        [3, 1, 0, 2],
        [3, 2, 1, 0],
    ];
    // Sign combinations: only the 3 non-zero positions (indices 1, 2, 3 in base) get signs
    for perm in &even_perms {
        // The zero is always at position perm[0] == 0, so we apply signs to positions 1,2,3
        for sign_mask in 0u8..8 {
            let s1 = if (sign_mask >> 0) & 1 == 1 { -1.0f32 } else { 1.0 };
            let s2 = if (sign_mask >> 1) & 1 == 1 { -1.0f32 } else { 1.0 };
            let s3 = if (sign_mask >> 2) & 1 == 1 { -1.0f32 } else { 1.0 };
            let signs = [0.0f32, s1, s2, s3]; // index 0 in base is zero, no sign needed
            let mut v = [0.0f32; 4];
            for (out_idx, &base_idx) in perm.iter().enumerate() {
                v[out_idx] = base[base_idx] * signs[base_idx];
            }
            vertices.push(v);
        }
    }
    // Count after Type C: 24 + 96 = 120

    vertices
}

/// Verify all 120 H4 vertices lie on the unit sphere (for testing/validation)
pub fn verify_h4_unit_sphere(vertices: &[[f32; 4]]) -> bool {
    vertices.iter().all(|v| {
        let norm_sq: f32 = v.iter().map(|x| x * x).sum();
        (norm_sq - 1.0).abs() < 1e-4
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_h4_vertex_count() {
        let vertices = generate_h4_vertices();
        assert_eq!(vertices.len(), H4_NUM_VERTICES, "Expected 120 vertices");
    }

    #[test]
    fn test_h4_vertices_on_unit_sphere() {
        let vertices = generate_h4_vertices();
        for (i, v) in vertices.iter().enumerate() {
            let norm_sq: f32 = v.iter().map(|x| x * x).sum();
            assert!(
                (norm_sq - 1.0).abs() < 1e-4,
                "Vertex {} has norm² = {:.6}, expected 1.0",
                i, norm_sq
            );
        }
    }

    #[test]
    fn test_h4_oracle_axis_aligned() {
        let vertices = generate_h4_vertices();

        // (1,0,0,0) is a Type A vertex — should find itself
        let query = [1.0f32, 0.0, 0.0, 0.0];
        let idx = H4Oracle::nearest_vertex_index(&query, &vertices);
        let result = vertices[idx as usize];
        let dist: f32 = result.iter().zip(query.iter()).map(|(a, b)| (a - b).powi(2)).sum();
        assert!(dist < 1e-5, "Oracle missed axis-aligned vertex, dist² = {}", dist);
    }

    #[test]
    fn test_h4_oracle_half_half() {
        let vertices = generate_h4_vertices();

        // (0.5, 0.5, 0.5, 0.5) is a Type B vertex
        let query = [0.5f32, 0.5, 0.5, 0.5];
        let idx = H4Oracle::nearest_vertex_index(&query, &vertices);
        let result = vertices[idx as usize];
        let dist: f32 = result.iter().zip(query.iter()).map(|(a, b)| (a - b).powi(2)).sum();
        assert!(dist < 1e-5, "Oracle missed Type B vertex, dist² = {}", dist);
    }

    #[test]
    fn test_hadamard4_involution() {
        let original = [1.0f32, 2.0, 3.0, 4.0];
        let mut data = original;

        hadamard4_inplace(&mut data);
        hadamard4_inplace(&mut data);

        // Two applications should recover original (normalized Hadamard is an involution)
        for i in 0..4 {
            assert!(
                (data[i] - original[i]).abs() < 1e-5,
                "Hadamard4 involution failed at index {}: got {}, expected {}",
                i, data[i], original[i]
            );
        }
    }

    #[test]
    fn test_hadamard4_orthogonality() {
        // H applied to standard basis vectors should produce orthogonal outputs
        let mut e0 = [1.0f32, 0.0, 0.0, 0.0];
        let mut e1 = [0.0f32, 1.0, 0.0, 0.0];
        hadamard4_inplace(&mut e0);
        hadamard4_inplace(&mut e1);
        let dot: f32 = e0.iter().zip(e1.iter()).map(|(a, b)| a * b).sum();
        assert!(dot.abs() < 1e-5, "Hadamard4 outputs not orthogonal, dot = {}", dot);
    }

    #[test]
    fn test_h4_codec_encode_decode_roundtrip() {
        let codec = H4Codec::new(768, true, 0xdeadbeef);

        let vector: Vec<f32> = (0..768).map(|i| (i as f32 * 0.01).cos()).collect();
        let encoded = codec.encode(&vector).unwrap();
        let decoded = codec.decode(&encoded);

        assert_eq!(decoded.len(), 768);
        assert_eq!(encoded.indices.len(), 192); // 768 / 4 blocks

        let mse: f32 = vector
            .iter()
            .zip(decoded.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / 768.0;

        assert!(mse < 0.5, "H4 codec MSE too high: {}", mse);
    }

    #[test]
    fn test_h4_codec_dimension_mismatch() {
        let codec = H4Codec::new(8, true, 42);
        let result = codec.encode(&[1.0, 2.0, 3.0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_h4_codec_memory_efficiency() {
        let codec = H4Codec::new(768, true, 42);

        let f32_bytes = 768 * 4; // 3072 bytes
        let h4_bytes = codec.bytes_per_vector();

        let ratio = f32_bytes as f32 / h4_bytes as f32;
        assert!(ratio > 3.0, "H4 compression ratio too low: {:.1}×", ratio);
    }

    #[test]
    fn test_h4_bits_per_dim() {
        let codec = H4Codec::new(4, false, 0);
        // log2(120) / 4 ≈ 6.907 / 4 ≈ 1.727 bits/dim
        let bpd = codec.bits_per_dim();
        assert!((bpd - 1.727).abs() < 0.01, "bits_per_dim = {}", bpd);
    }

    #[test]
    fn test_h4_encoded_vector_size() {
        let codec = H4Codec::new(16, false, 0);
        let v: Vec<f32> = vec![0.5; 16];
        let encoded = codec.encode(&v).unwrap();
        // 4 blocks × 1 byte + 4 bytes scale = 8 bytes
        assert_eq!(encoded.size_bytes(), 8);
    }
}
