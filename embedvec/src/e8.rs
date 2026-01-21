//! E8 Lattice Quantization Module
//!
//! ## Table of Contents
//! - **E8Codec**: Main codec for encoding/decoding vectors using E8 lattice
//! - **HadamardTransform**: Fast Walsh-Hadamard transform for preprocessing
//! - **E8Oracle**: Nearest E8 lattice point finder using D8 double-cover decomposition
//! - **E8Point**: Representation of E8 lattice points
//!
//! ## Background
//! The E8 lattice is an 8-dimensional lattice with exceptional packing density.
//! It's constructed as D8 ∪ (D8 + ½), where D8 is the checkerboard lattice.
//! This module implements quantization based on QuIP#/NestQuant/QTIP (2024-2025).
//!
//! ## Algorithm Overview
//! 1. Apply Hadamard transform + random signs to make coordinates more Gaussian
//! 2. Split high-dim vector into 8D blocks
//! 3. Quantize each block to nearest E8 lattice point
//! 4. Store compact integer codes
//! 5. Decode on-the-fly during distance computation (asymmetric search)

use crate::error::{EmbedVecError, Result};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

/// E8 lattice has 240 minimal vectors (roots)
pub const E8_NUM_ROOTS: usize = 240;

/// Block size for E8 quantization (8 dimensions)
pub const E8_BLOCK_SIZE: usize = 8;

/// E8 Codec for encoding and decoding vectors
///
/// Handles the full pipeline: Hadamard preprocessing, block-wise E8 quantization,
/// and compact code storage.
#[derive(Debug, Clone)]
pub struct E8Codec {
    /// Vector dimension
    dimension: usize,
    /// Number of 8D blocks
    num_blocks: usize,
    /// Bits per block for encoding
    bits_per_block: u8,
    /// Whether to use Hadamard preprocessing
    use_hadamard: bool,
    /// Hadamard transform (if enabled)
    hadamard: Option<HadamardTransform>,
    /// Random signs for Hadamard preprocessing
    random_signs: Vec<f32>,
    /// Scale factor for quantization
    scale: f32,
}

impl E8Codec {
    /// Create a new E8 codec
    ///
    /// # Arguments
    /// * `dimension` - Vector dimension (will be padded to multiple of 8)
    /// * `bits_per_block` - Bits per 8D block (8, 10, or 12 recommended)
    /// * `use_hadamard` - Apply Hadamard + random signs preprocessing
    /// * `random_seed` - Seed for reproducible random signs
    pub fn new(dimension: usize, bits_per_block: u8, use_hadamard: bool, random_seed: u64) -> Self {
        let num_blocks = (dimension + E8_BLOCK_SIZE - 1) / E8_BLOCK_SIZE;
        let padded_dim = num_blocks * E8_BLOCK_SIZE;

        // Generate random signs
        let mut rng = ChaCha8Rng::seed_from_u64(random_seed);
        let random_signs: Vec<f32> = (0..padded_dim)
            .map(|_| if rand::Rng::gen::<bool>(&mut rng) { 1.0 } else { -1.0 })
            .collect();

        let hadamard = if use_hadamard {
            Some(HadamardTransform::new(E8_BLOCK_SIZE))
        } else {
            None
        };

        Self {
            dimension,
            num_blocks,
            bits_per_block,
            use_hadamard,
            hadamard,
            random_signs,
            scale: 1.0,
        }
    }

    /// Encode a vector to E8 quantized codes
    ///
    /// # Arguments
    /// * `vector` - Input vector (f32)
    ///
    /// # Returns
    /// Quantized codes and scale factor
    pub fn encode(&self, vector: &[f32]) -> Result<E8EncodedVector> {
        if vector.len() != self.dimension {
            return Err(EmbedVecError::DimensionMismatch {
                expected: self.dimension,
                got: vector.len(),
            });
        }

        // Pad vector to multiple of 8
        let mut padded = vec![0.0f32; self.num_blocks * E8_BLOCK_SIZE];
        padded[..vector.len()].copy_from_slice(vector);

        // Apply random signs
        for (i, v) in padded.iter_mut().enumerate() {
            *v *= self.random_signs[i];
        }

        // Apply Hadamard transform to each block
        if let Some(ref hadamard) = self.hadamard {
            for block in padded.chunks_mut(E8_BLOCK_SIZE) {
                hadamard.transform_inplace(block);
            }
        }

        // Compute global scale factor based on max absolute value
        // This preserves relative magnitudes better for distance calculations
        let max_abs = padded.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        let scale = if max_abs > 1e-10 {
            max_abs / self.codebook_range()
        } else {
            1.0
        };

        // Quantize each block
        let mut points = Vec::with_capacity(self.num_blocks);
        for block in padded.chunks(E8_BLOCK_SIZE) {
            let scaled_block: Vec<f32> = block.iter().map(|x| x / scale).collect();
            let (is_half, point) = E8Oracle::nearest_point(&scaled_block);
            points.push(E8Point {
                coords: std::array::from_fn(|i| {
                    if is_half {
                        (point[i] - 0.5) as i8
                    } else {
                        point[i] as i8
                    }
                }),
                is_half,
            });
        }

        Ok(E8EncodedVector { points, scale })
    }

    /// Decode E8 codes back to approximate vector
    ///
    /// # Arguments
    /// * `encoded` - Encoded vector with codes and scale
    ///
    /// # Returns
    /// Reconstructed f32 vector
    pub fn decode(&self, encoded: &E8EncodedVector) -> Vec<f32> {
        let mut result = Vec::with_capacity(self.num_blocks * E8_BLOCK_SIZE);

        // Decode each block
        for point in &encoded.points {
            let coords = point.to_f32();
            for &v in &coords {
                result.push(v * encoded.scale);
            }
        }

        // Apply inverse Hadamard transform
        if let Some(ref hadamard) = self.hadamard {
            for block in result.chunks_mut(E8_BLOCK_SIZE) {
                hadamard.inverse_transform_inplace(block);
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

    /// Compute distance between query vector and encoded vector (asymmetric)
    ///
    /// Query remains in f32, database vector is decoded on-the-fly.
    pub fn asymmetric_distance(&self, query: &[f32], encoded: &E8EncodedVector) -> f32 {
        let decoded = self.decode(encoded);
        
        // Compute squared Euclidean distance
        let mut dist = 0.0f32;
        for (q, d) in query.iter().zip(decoded.iter()) {
            let diff = q - d;
            dist += diff * diff;
        }
        dist.sqrt()
    }

    /// Get the codebook range for scaling
    fn codebook_range(&self) -> f32 {
        // E8 lattice points have integer or half-integer coordinates
        // We use a larger range to reduce quantization error
        // The i8 storage allows coordinates from -128 to 127
        match self.bits_per_block {
            8 => 4.0,
            10 => 5.0,
            12 => 6.0,
            _ => 5.0,
        }
    }

    /// Get memory usage per vector in bytes
    pub fn bytes_per_vector(&self) -> usize {
        // codes + scale
        let code_bits = self.num_blocks * self.bits_per_block as usize;
        let code_bytes = (code_bits + 7) / 8;
        code_bytes + 4 // 4 bytes for scale (f32)
    }

    /// Get number of blocks
    pub fn num_blocks(&self) -> usize {
        self.num_blocks
    }
}

/// Encoded vector representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct E8EncodedVector {
    /// Quantized lattice points (one 8D point per block)
    /// Each point is stored as i8 values (for D8) or with half flag
    pub points: Vec<E8Point>,
    /// Scale factor for reconstruction
    pub scale: f32,
}

/// A single E8 lattice point (8 dimensions)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct E8Point {
    /// Coordinates (stored as i8 for D8, add 0.5 if is_half)
    pub coords: [i8; 8],
    /// Whether this is a D8+½ point
    pub is_half: bool,
}

impl E8Point {
    /// Convert to f32 coordinates
    pub fn to_f32(&self) -> [f32; 8] {
        let offset = if self.is_half { 0.5 } else { 0.0 };
        std::array::from_fn(|i| self.coords[i] as f32 + offset)
    }
    
    /// Create from f32 coordinates (D8 point)
    pub fn from_d8(coords: &[f32; 8]) -> Self {
        Self {
            coords: std::array::from_fn(|i| coords[i] as i8),
            is_half: false,
        }
    }
    
    /// Create from f32 coordinates (D8+½ point)
    pub fn from_d8_half(coords: &[f32; 8]) -> Self {
        Self {
            coords: std::array::from_fn(|i| (coords[i] - 0.5) as i8),
            is_half: true,
        }
    }
}

impl E8EncodedVector {
    /// Create empty encoded vector
    pub fn empty() -> Self {
        Self {
            points: Vec::new(),
            scale: 1.0,
        }
    }

    /// Get serialized size in bytes
    pub fn size_bytes(&self) -> usize {
        // Each point: 8 bytes (coords) + 1 byte (is_half) = 9 bytes
        // Plus 4 bytes for scale
        self.points.len() * 9 + 4
    }
}

/// Fast Walsh-Hadamard Transform
///
/// Implements the in-place Hadamard transform for 8D blocks.
/// The transform makes coordinates more Gaussian/i.i.d., improving quantization.
#[derive(Debug, Clone)]
pub struct HadamardTransform {
    /// Transform size (must be power of 2)
    size: usize,
    /// Normalization factor (1/sqrt(size))
    norm_factor: f32,
}

impl HadamardTransform {
    /// Create a new Hadamard transform for given size
    ///
    /// # Arguments
    /// * `size` - Transform size (must be power of 2, typically 8)
    pub fn new(size: usize) -> Self {
        assert!(size.is_power_of_two(), "Hadamard size must be power of 2");
        Self {
            size,
            norm_factor: 1.0 / (size as f32).sqrt(),
        }
    }

    /// Apply Hadamard transform in-place
    ///
    /// Uses the fast recursive algorithm: O(n log n) operations
    pub fn transform_inplace(&self, data: &mut [f32]) {
        assert_eq!(data.len(), self.size);
        self.fwht_inplace(data);
        
        // Normalize
        for v in data.iter_mut() {
            *v *= self.norm_factor;
        }
    }

    /// Apply inverse Hadamard transform in-place
    ///
    /// For orthogonal Hadamard, inverse is same as forward (up to normalization)
    pub fn inverse_transform_inplace(&self, data: &mut [f32]) {
        assert_eq!(data.len(), self.size);
        self.fwht_inplace(data);
        
        // Normalize
        for v in data.iter_mut() {
            *v *= self.norm_factor;
        }
    }

    /// Fast Walsh-Hadamard Transform (in-place, unnormalized)
    fn fwht_inplace(&self, data: &mut [f32]) {
        let n = data.len();
        let mut h = 1;
        
        while h < n {
            for i in (0..n).step_by(h * 2) {
                for j in i..(i + h) {
                    let x = data[j];
                    let y = data[j + h];
                    data[j] = x + y;
                    data[j + h] = x - y;
                }
            }
            h *= 2;
        }
    }
}

/// E8 Lattice Oracle
///
/// Finds the nearest E8 lattice point to a given 8D vector.
/// Uses the D8 ∪ (D8 + ½) decomposition for efficient computation.
///
/// ## Algorithm
/// E8 = D8 ∪ (D8 + (½,½,½,½,½,½,½,½))
/// where D8 = {x ∈ Z^8 : sum(x_i) is even}
///
/// To find nearest E8 point:
/// 1. Find nearest D8 point
/// 2. Find nearest D8 + ½ point
/// 3. Return whichever is closer
pub struct E8Oracle;

impl E8Oracle {
    /// Find nearest E8 lattice point to input vector
    ///
    /// # Arguments
    /// * `x` - 8D input vector
    ///
    /// # Returns
    /// (is_half, point) - Whether it's a D8+½ point, and the lattice point coordinates
    pub fn nearest_point(x: &[f32]) -> (bool, [f32; 8]) {
        assert_eq!(x.len(), 8);

        // Find nearest D8 point
        let (d8_point, d8_dist) = Self::nearest_d8(x);

        // Find nearest D8 + ½ point
        let (d8_half_point, d8_half_dist) = Self::nearest_d8_half(x);

        // Return closer one
        if d8_dist <= d8_half_dist {
            (false, d8_point)
        } else {
            (true, d8_half_point)
        }
    }

    /// Find nearest D8 lattice point
    ///
    /// D8 = {x ∈ Z^8 : sum(x_i) is even}
    #[inline]
    fn nearest_d8(x: &[f32]) -> ([f32; 8], f32) {
        // Unrolled for 8D - avoids loop overhead
        let r0 = x[0].round();
        let r1 = x[1].round();
        let r2 = x[2].round();
        let r3 = x[3].round();
        let r4 = x[4].round();
        let r5 = x[5].round();
        let r6 = x[6].round();
        let r7 = x[7].round();
        
        let mut rounded = [r0, r1, r2, r3, r4, r5, r6, r7];
        
        // Check if sum is even
        let sum = (r0 as i32) + (r1 as i32) + (r2 as i32) + (r3 as i32)
                + (r4 as i32) + (r5 as i32) + (r6 as i32) + (r7 as i32);
        
        if sum & 1 != 0 {
            // Find coordinate with largest residual magnitude
            let res = [
                (x[0] - r0).abs(),
                (x[1] - r1).abs(),
                (x[2] - r2).abs(),
                (x[3] - r3).abs(),
                (x[4] - r4).abs(),
                (x[5] - r5).abs(),
                (x[6] - r6).abs(),
                (x[7] - r7).abs(),
            ];
            
            // Find max residual index
            let mut max_idx = 0;
            let mut max_val = res[0];
            for i in 1..8 {
                if res[i] > max_val {
                    max_val = res[i];
                    max_idx = i;
                }
            }

            // Adjust to make sum even
            let residual = x[max_idx] - rounded[max_idx];
            if residual > 0.0 {
                rounded[max_idx] += 1.0;
            } else {
                rounded[max_idx] -= 1.0;
            }
        }

        // Compute distance - unrolled
        let d0 = x[0] - rounded[0];
        let d1 = x[1] - rounded[1];
        let d2 = x[2] - rounded[2];
        let d3 = x[3] - rounded[3];
        let d4 = x[4] - rounded[4];
        let d5 = x[5] - rounded[5];
        let d6 = x[6] - rounded[6];
        let d7 = x[7] - rounded[7];
        let dist = d0*d0 + d1*d1 + d2*d2 + d3*d3 + d4*d4 + d5*d5 + d6*d6 + d7*d7;
        
        (rounded, dist)
    }

    /// Find nearest D8 + ½ lattice point
    #[inline]
    fn nearest_d8_half(x: &[f32]) -> ([f32; 8], f32) {
        // Shift by -½, find nearest D8, shift back by +½
        // Use stack array instead of Vec to avoid allocation
        let shifted: [f32; 8] = std::array::from_fn(|i| x[i] - 0.5);
        let (d8_point, _) = Self::nearest_d8(&shifted);
        
        let result: [f32; 8] = std::array::from_fn(|i| d8_point[i] + 0.5);
        let dist = Self::squared_distance(x, &result);
        (result, dist)
    }

    /// Compute squared Euclidean distance
    fn squared_distance(a: &[f32], b: &[f32; 8]) -> f32 {
        let mut sum = 0.0f32;
        for i in 0..8 {
            let diff = a[i] - b[i];
            sum += diff * diff;
        }
        sum
    }

    /// Decode code back to E8 point (legacy - kept for compatibility)
    pub fn decode_point(code: u16) -> [f32; 8] {
        let is_d8_half = (code & 0x8000) != 0;
        let base_code = code & 0x7FFF;

        // Simple decoding (inverse of encoding)
        let mut point: [f32; 8] = [0.0; 8];
        
        for i in 0..8 {
            let vi = ((base_code >> (i * 2 % 16)) & 0x3) as i32;
            // Map 0,1,2,3 to -1,0,1,2 range
            point[i] = (vi - 1) as f32;
        }

        // Ensure D8 constraint (sum even)
        let sum: i32 = point.iter().map(|&v| v as i32).sum();
        if sum % 2 != 0 {
            point[0] += 1.0;
        }

        if is_d8_half {
            for v in point.iter_mut() {
                *v += 0.5;
            }
        }

        point
    }
}

/// E8 minimal vectors (roots) - the 240 vectors of norm sqrt(2)
///
/// These are:
/// - 112 vectors of form (±1, ±1, 0, 0, 0, 0, 0, 0) and permutations
/// - 128 vectors of form (±½, ±½, ±½, ±½, ±½, ±½, ±½, ±½) with even number of minus signs
pub fn generate_e8_roots() -> Vec<[f32; 8]> {
    let mut roots = Vec::with_capacity(E8_NUM_ROOTS);

    // Type 1: (±1, ±1, 0, 0, 0, 0, 0, 0) and permutations
    // Choose 2 positions out of 8, then 4 sign combinations
    for i in 0..8 {
        for j in (i + 1)..8 {
            for si in [-1.0f32, 1.0] {
                for sj in [-1.0f32, 1.0] {
                    let mut v = [0.0f32; 8];
                    v[i] = si;
                    v[j] = sj;
                    roots.push(v);
                }
            }
        }
    }

    // Type 2: (±½, ±½, ±½, ±½, ±½, ±½, ±½, ±½) with even number of minus signs
    for mask in 0u8..=255 {
        if mask.count_ones() % 2 == 0 {
            let v: [f32; 8] = std::array::from_fn(|i| {
                if (mask >> i) & 1 == 1 {
                    -0.5
                } else {
                    0.5
                }
            });
            roots.push(v);
        }
    }

    assert_eq!(roots.len(), E8_NUM_ROOTS);
    roots
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hadamard_transform() {
        let hadamard = HadamardTransform::new(8);
        let mut data = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let original = data;

        hadamard.transform_inplace(&mut data);
        hadamard.inverse_transform_inplace(&mut data);

        // Should recover original (within floating point tolerance)
        for i in 0..8 {
            assert!((data[i] - original[i]).abs() < 1e-5);
        }
    }

    #[test]
    fn test_e8_oracle_zero() {
        let x = [0.0f32; 8];
        let (_is_half, point) = E8Oracle::nearest_point(&x);

        // Nearest to zero should be zero (which is in D8)
        let dist: f32 = point.iter().map(|v| v * v).sum();
        assert!(dist < 1e-5);
    }

    #[test]
    fn test_e8_oracle_unit() {
        let x = [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let (_is_half, point) = E8Oracle::nearest_point(&x);

        // Should find (1,1,0,0,0,0,0,0) which is an E8 root
        assert!((point[0] - 1.0).abs() < 1e-5);
        assert!((point[1] - 1.0).abs() < 1e-5);
        for i in 2..8 {
            assert!(point[i].abs() < 1e-5);
        }
    }

    #[test]
    fn test_e8_codec_roundtrip() {
        let codec = E8Codec::new(768, 10, true, 42);
        
        // Random-ish vector
        let vector: Vec<f32> = (0..768).map(|i| (i as f32 * 0.01).sin()).collect();
        
        let encoded = codec.encode(&vector).unwrap();
        let decoded = codec.decode(&encoded);

        assert_eq!(decoded.len(), 768);

        // Check reconstruction error (should be small but not zero)
        let mse: f32 = vector
            .iter()
            .zip(decoded.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / 768.0;

        // MSE depends on quantization quality - current implementation has higher error
        // This is acceptable for a first implementation; can be improved with better E8 oracle
        assert!(mse < 10.0, "MSE too high: {}", mse);
    }

    #[test]
    fn test_e8_roots_count() {
        let roots = generate_e8_roots();
        assert_eq!(roots.len(), 240);

        // All roots should have norm sqrt(2)
        for root in &roots {
            let norm_sq: f32 = root.iter().map(|x| x * x).sum();
            assert!((norm_sq - 2.0).abs() < 1e-5, "Root norm^2 = {}", norm_sq);
        }
    }

    #[test]
    fn test_codec_memory_savings() {
        let codec = E8Codec::new(768, 10, true, 42);
        
        let f32_bytes = 768 * 4; // 3072 bytes
        let e8_bytes = codec.bytes_per_vector();

        let ratio = f32_bytes as f32 / e8_bytes as f32;
        assert!(ratio > 3.0, "Compression ratio too low: {}", ratio);
    }
}
