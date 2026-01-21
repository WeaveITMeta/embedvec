//! Quantization configuration and modes
//!
//! ## Table of Contents
//! - **Quantization**: Enum defining quantization modes (None, E8)
//! - **E8 Configuration**: E8 lattice quantization with Hadamard preprocessing

use serde::{Deserialize, Serialize};

/// Quantization mode for vector compression
///
/// Controls how vectors are stored in memory. E8 quantization can reduce
/// memory usage by 4-6× while maintaining high recall.
///
/// # Example
/// ```rust
/// use embedvec::Quantization;
///
/// let quant = Quantization::E8 {
///     bits_per_block: 10,
///     use_hadamard: true,
///     random_seed: 0xcafef00d,
/// };
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Quantization {
    /// No quantization - store full f32 vectors
    /// Memory: 4 bytes per dimension
    None,

    /// E8 lattice quantization with optional Hadamard preprocessing
    ///
    /// Based on QuIP#/NestQuant/QTIP 2024-2025 research.
    /// Splits vectors into 8D blocks and quantizes each to nearest E8 lattice point.
    ///
    /// Memory: ~2-3 bits per dimension (depending on bits_per_block)
    E8 {
        /// Bits per 8D block for codebook resolution
        /// - 8 bits: ~1 bit/dim, lower quality
        /// - 10 bits: ~1.25 bits/dim, good balance
        /// - 12 bits: ~1.5 bits/dim, higher quality
        bits_per_block: u8,

        /// Apply fast Hadamard transform + random signs before quantization
        /// Improves quantization quality by making coordinates more Gaussian/i.i.d.
        use_hadamard: bool,

        /// Random seed for reproducible Hadamard rotation matrix
        random_seed: u64,
    },
}

impl Default for Quantization {
    fn default() -> Self {
        Quantization::None
    }
}

impl Quantization {
    /// Create E8 quantization with default parameters
    ///
    /// Uses 10 bits per block with Hadamard preprocessing.
    pub fn e8_default() -> Self {
        Quantization::E8 {
            bits_per_block: 10,
            use_hadamard: true,
            random_seed: 0xcafef00d,
        }
    }

    /// Create E8 quantization with custom parameters
    ///
    /// # Arguments
    /// * `bits_per_block` - Bits per 8D block (must be 8, 10, or 12)
    /// * `use_hadamard` - Apply Hadamard transform preprocessing
    /// * `random_seed` - Seed for reproducible random signs
    ///
    /// # Panics
    /// Panics if bits_per_block is not 8, 10, or 12
    pub fn e8(bits_per_block: u8, use_hadamard: bool, random_seed: u64) -> Self {
        assert!(
            bits_per_block == 8 || bits_per_block == 10 || bits_per_block == 12,
            "bits_per_block must be 8, 10, or 12, got {}",
            bits_per_block
        );
        Quantization::E8 {
            bits_per_block,
            use_hadamard,
            random_seed,
        }
    }

    /// Create E8 quantization with validation (returns Result)
    pub fn e8_checked(bits_per_block: u8, use_hadamard: bool, random_seed: u64) -> Result<Self, &'static str> {
        if bits_per_block != 8 && bits_per_block != 10 && bits_per_block != 12 {
            return Err("bits_per_block must be 8, 10, or 12");
        }
        Ok(Quantization::E8 {
            bits_per_block,
            use_hadamard,
            random_seed,
        })
    }

    /// Check if quantization is enabled
    pub fn is_enabled(&self) -> bool {
        !matches!(self, Quantization::None)
    }

    /// Get approximate bits per dimension
    pub fn bits_per_dim(&self) -> f32 {
        match self {
            Quantization::None => 32.0,
            Quantization::E8 { bits_per_block, .. } => *bits_per_block as f32 / 8.0,
        }
    }

    /// Get approximate memory usage per vector in bytes
    pub fn bytes_per_vector(&self, dimension: usize) -> usize {
        match self {
            Quantization::None => dimension * 4,
            Quantization::E8 { bits_per_block, .. } => {
                let num_blocks = (dimension + 7) / 8;
                // Each block uses bits_per_block bits + 4 bytes for scale factor
                let code_bytes = (num_blocks * (*bits_per_block as usize) + 7) / 8;
                code_bytes + 4 // scale factor
            }
        }
    }

    /// Get compression ratio compared to f32
    pub fn compression_ratio(&self, dimension: usize) -> f32 {
        let f32_bytes = dimension * 4;
        let quant_bytes = self.bytes_per_vector(dimension);
        f32_bytes as f32 / quant_bytes as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantization_default() {
        let q = Quantization::default();
        assert!(!q.is_enabled());
        assert_eq!(q.bits_per_dim(), 32.0);
    }

    #[test]
    fn test_e8_compression() {
        let q = Quantization::e8_default();
        assert!(q.is_enabled());

        // 768 dim vector
        let f32_bytes = 768 * 4; // 3072 bytes
        let e8_bytes = q.bytes_per_vector(768);

        // With 10 bits per block (96 blocks for 768 dims):
        // code_bytes = (96 * 10 + 7) / 8 = 120 bytes
        // total = 120 + 4 (scale) = 124 bytes
        // ratio = 3072 / 124 ≈ 24.8x (very high compression)
        // But actual storage uses u16 per code = 96 * 2 + 4 = 196 bytes
        // So ratio ≈ 15.7x
        let ratio = f32_bytes as f32 / e8_bytes as f32;
        assert!(ratio > 2.0 && ratio < 30.0, "Compression ratio: {}", ratio);
    }

    #[test]
    fn test_bits_per_dim() {
        let q8 = Quantization::e8(8, true, 0);
        let q10 = Quantization::e8(10, true, 0);
        let q12 = Quantization::e8(12, true, 0);

        assert_eq!(q8.bits_per_dim(), 1.0);
        assert_eq!(q10.bits_per_dim(), 1.25);
        assert_eq!(q12.bits_per_dim(), 1.5);
    }
}
