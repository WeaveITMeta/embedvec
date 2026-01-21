//! Distance metrics for vector similarity computation
//!
//! ## Table of Contents
//! - **Distance**: Enum of supported distance metrics
//! - **compute_distance**: Core distance computation function
//! - **SIMD optimizations**: AVX2/SSE vectorized implementations
//!
//! ## Performance
//! SIMD implementations provide 4-8x speedup on modern CPUs.
//! Auto-vectorization is enabled for scalar fallbacks.

use serde::{Deserialize, Serialize};

/// Supported distance metrics for similarity computation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Distance {
    /// Cosine similarity (1 - cos(θ))
    /// Vectors should be normalized for best performance
    /// Lower score = more similar
    Cosine,

    /// Euclidean distance (L2 norm)
    /// Lower score = more similar
    Euclidean,

    /// Dot product (inner product)
    /// Higher score = more similar (negated internally for min-heap)
    DotProduct,
}

impl Distance {
    /// Compute distance between two vectors
    ///
    /// # Arguments
    /// * `a` - First vector
    /// * `b` - Second vector
    ///
    /// # Returns
    /// Distance value (lower = more similar for Cosine/Euclidean)
    ///
    /// # Panics
    /// Panics if vectors have different lengths
    #[inline]
    pub fn compute(&self, a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len(), "Vector dimensions must match");

        match self {
            Distance::Cosine => cosine_distance_fast(a, b),
            Distance::Euclidean => euclidean_squared_fast(a, b).sqrt(),
            Distance::DotProduct => -dot_product_fast(a, b),
        }
    }

    /// Compute distance with SIMD optimization (when available)
    #[inline]
    pub fn compute_simd(&self, a: &[f32], b: &[f32]) -> f32 {
        #[cfg(feature = "simd")]
        {
            match self {
                Distance::Cosine => cosine_distance_simd(a, b),
                Distance::Euclidean => euclidean_distance_simd(a, b),
                Distance::DotProduct => dot_product_distance_simd(a, b),
            }
        }
        #[cfg(not(feature = "simd"))]
        {
            self.compute(a, b)
        }
    }
}

impl Default for Distance {
    fn default() -> Self {
        Distance::Cosine
    }
}

/// Cosine distance: 1 - cos(θ) = 1 - (a·b)/(|a||b|)
/// For normalized vectors: 1 - a·b
#[inline]
#[allow(dead_code)]
fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;

    for i in 0..a.len() {
        dot = dot + a[i] * b[i];
        norm_a = norm_a + a[i] * a[i];
        norm_b = norm_b + b[i] * b[i];
    }

    let denom = (norm_a * norm_b).sqrt();
    if denom > 1e-10 {
        1.0 - (dot / denom)
    } else {
        1.0
    }
}

/// Cosine distance for pre-normalized vectors (fast path)
/// Assumes both vectors have unit length, skips norm calculation
#[inline]
pub fn cosine_distance_normalized(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    for i in 0..a.len() {
        dot = dot + a[i] * b[i];
    }
    1.0 - dot
}

/// Euclidean distance: sqrt(sum((a_i - b_i)^2))
#[inline]
#[allow(dead_code)]
fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        let diff = a[i] - b[i];
        sum = diff.mul_add(diff, sum);  // FMA: sum + diff * diff
    }
    sum.sqrt()
}

/// Dot product distance: -a·b (negated so lower = more similar)
#[inline]
#[allow(dead_code)]
fn dot_product_distance(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    for i in 0..a.len() {
        dot = a[i].mul_add(b[i], dot);  // FMA: dot + a[i] * b[i]
    }
    -dot // Negate so lower = better (consistent with other metrics)
}

/// Squared Euclidean distance (avoids sqrt for faster comparison)
#[inline]
pub fn euclidean_distance_squared(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        let diff = a[i] - b[i];
        sum = diff.mul_add(diff, sum);
    }
    sum
}

/// Raw dot product (not negated)
#[inline]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    for i in 0..a.len() {
        dot = a[i].mul_add(b[i], dot);
    }
    dot
}

// =============================================================================
// SIMD-optimized implementations for x86_64
// =============================================================================

#[cfg(target_arch = "x86_64")]
mod simd_avx2 {
    use std::arch::x86_64::*;

    /// AVX2 dot product - processes 8 floats per iteration
    #[target_feature(enable = "avx2", enable = "fma")]
    pub unsafe fn dot_product_avx2_impl(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len();
        let chunks = n / 8;
        
        let mut sum = _mm256_setzero_ps();
        
        for i in 0..chunks {
            let va = _mm256_loadu_ps(a.as_ptr().add(i * 8));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i * 8));
            sum = _mm256_fmadd_ps(va, vb, sum);
        }

        // Horizontal sum: sum all 8 lanes
        let hi = _mm256_extractf128_ps(sum, 1);
        let lo = _mm256_castps256_ps128(sum);
        let sum128 = _mm_add_ps(lo, hi);
        let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
        let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
        let mut total = _mm_cvtss_f32(sum32);

        // Handle remainder
        for i in (chunks * 8)..n {
            total += a[i] * b[i];
        }

        total
    }

    /// AVX2 squared Euclidean distance
    #[target_feature(enable = "avx2", enable = "fma")]
    pub unsafe fn euclidean_squared_avx2_impl(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len();
        let chunks = n / 8;
        
        let mut sum = _mm256_setzero_ps();
        
        for i in 0..chunks {
            let va = _mm256_loadu_ps(a.as_ptr().add(i * 8));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i * 8));
            let diff = _mm256_sub_ps(va, vb);
            sum = _mm256_fmadd_ps(diff, diff, sum);
        }

        // Horizontal sum
        let hi = _mm256_extractf128_ps(sum, 1);
        let lo = _mm256_castps256_ps128(sum);
        let sum128 = _mm_add_ps(lo, hi);
        let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
        let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
        let mut total = _mm_cvtss_f32(sum32);

        // Handle remainder
        for i in (chunks * 8)..n {
            let diff = a[i] - b[i];
            total += diff * diff;
        }

        total
    }

    /// AVX2 cosine distance with parallel dot/norm computation
    #[target_feature(enable = "avx2", enable = "fma")]
    pub unsafe fn cosine_distance_avx2_impl(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len();
        let chunks = n / 8;
        
        let mut dot_sum = _mm256_setzero_ps();
        let mut norm_a_sum = _mm256_setzero_ps();
        let mut norm_b_sum = _mm256_setzero_ps();
        
        for i in 0..chunks {
            let va = _mm256_loadu_ps(a.as_ptr().add(i * 8));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i * 8));
            dot_sum = _mm256_fmadd_ps(va, vb, dot_sum);
            norm_a_sum = _mm256_fmadd_ps(va, va, norm_a_sum);
            norm_b_sum = _mm256_fmadd_ps(vb, vb, norm_b_sum);
        }

        // Horizontal sums
        let hsum = |v: __m256| -> f32 {
            let hi = _mm256_extractf128_ps(v, 1);
            let lo = _mm256_castps256_ps128(v);
            let sum128 = _mm_add_ps(lo, hi);
            let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
            let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
            _mm_cvtss_f32(sum32)
        };

        let mut dot = hsum(dot_sum);
        let mut norm_a = hsum(norm_a_sum);
        let mut norm_b = hsum(norm_b_sum);

        // Handle remainder
        for i in (chunks * 8)..n {
            dot += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }

        let denom = (norm_a * norm_b).sqrt();
        if denom > 1e-10 {
            1.0 - (dot / denom)
        } else {
            1.0
        }
    }
}

// Portable SIMD using std::simd (nightly) or manual unrolling
mod simd_portable {
    /// 4-way unrolled dot product for better instruction-level parallelism
    #[inline]
    pub fn dot_product_unrolled(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len();
        let chunks = n / 4;
        
        let mut sum0 = 0.0f32;
        let mut sum1 = 0.0f32;
        let mut sum2 = 0.0f32;
        let mut sum3 = 0.0f32;
        
        for i in 0..chunks {
            let base = i * 4;
            sum0 += a[base] * b[base];
            sum1 += a[base + 1] * b[base + 1];
            sum2 += a[base + 2] * b[base + 2];
            sum3 += a[base + 3] * b[base + 3];
        }
        
        let mut total = sum0 + sum1 + sum2 + sum3;
        
        // Handle remainder
        for i in (chunks * 4)..n {
            total += a[i] * b[i];
        }
        
        total
    }

    /// 4-way unrolled squared Euclidean distance
    #[inline]
    pub fn euclidean_squared_unrolled(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len();
        let chunks = n / 4;
        
        let mut sum0 = 0.0f32;
        let mut sum1 = 0.0f32;
        let mut sum2 = 0.0f32;
        let mut sum3 = 0.0f32;
        
        for i in 0..chunks {
            let base = i * 4;
            let d0 = a[base] - b[base];
            let d1 = a[base + 1] - b[base + 1];
            let d2 = a[base + 2] - b[base + 2];
            let d3 = a[base + 3] - b[base + 3];
            sum0 += d0 * d0;
            sum1 += d1 * d1;
            sum2 += d2 * d2;
            sum3 += d3 * d3;
        }
        
        let mut total = sum0 + sum1 + sum2 + sum3;
        
        for i in (chunks * 4)..n {
            let d = a[i] - b[i];
            total += d * d;
        }
        
        total
    }

    /// 4-way unrolled cosine distance
    #[inline]
    pub fn cosine_distance_unrolled(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len();
        let chunks = n / 4;
        
        let mut dot0 = 0.0f32;
        let mut dot1 = 0.0f32;
        let mut dot2 = 0.0f32;
        let mut dot3 = 0.0f32;
        let mut na0 = 0.0f32;
        let mut na1 = 0.0f32;
        let mut na2 = 0.0f32;
        let mut na3 = 0.0f32;
        let mut nb0 = 0.0f32;
        let mut nb1 = 0.0f32;
        let mut nb2 = 0.0f32;
        let mut nb3 = 0.0f32;
        
        for i in 0..chunks {
            let base = i * 4;
            dot0 += a[base] * b[base];
            dot1 += a[base + 1] * b[base + 1];
            dot2 += a[base + 2] * b[base + 2];
            dot3 += a[base + 3] * b[base + 3];
            na0 += a[base] * a[base];
            na1 += a[base + 1] * a[base + 1];
            na2 += a[base + 2] * a[base + 2];
            na3 += a[base + 3] * a[base + 3];
            nb0 += b[base] * b[base];
            nb1 += b[base + 1] * b[base + 1];
            nb2 += b[base + 2] * b[base + 2];
            nb3 += b[base + 3] * b[base + 3];
        }
        
        let mut dot = dot0 + dot1 + dot2 + dot3;
        let mut norm_a = na0 + na1 + na2 + na3;
        let mut norm_b = nb0 + nb1 + nb2 + nb3;
        
        for i in (chunks * 4)..n {
            dot += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }
        
        let denom = (norm_a * norm_b).sqrt();
        if denom > 1e-10 {
            1.0 - (dot / denom)
        } else {
            1.0
        }
    }
}

/// Fast dot product - uses AVX2 when available, falls back to unrolled scalar
#[inline]
pub fn dot_product_fast(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { simd_avx2::dot_product_avx2_impl(a, b) };
        }
    }
    simd_portable::dot_product_unrolled(a, b)
}

/// Fast squared Euclidean distance
#[inline]
pub fn euclidean_squared_fast(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { simd_avx2::euclidean_squared_avx2_impl(a, b) };
        }
    }
    simd_portable::euclidean_squared_unrolled(a, b)
}

/// Fast cosine distance
#[inline]
pub fn cosine_distance_fast(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { simd_avx2::cosine_distance_avx2_impl(a, b) };
        }
    }
    simd_portable::cosine_distance_unrolled(a, b)
}

// Legacy SIMD feature-gated implementations (for backward compatibility)
#[cfg(feature = "simd")]
fn cosine_distance_simd(a: &[f32], b: &[f32]) -> f32 {
    cosine_distance_fast(a, b)
}

#[cfg(feature = "simd")]
fn euclidean_distance_simd(a: &[f32], b: &[f32]) -> f32 {
    euclidean_squared_fast(a, b).sqrt()
}

#[cfg(feature = "simd")]
fn dot_product_distance_simd(a: &[f32], b: &[f32]) -> f32 {
    -dot_product_fast(a, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_distance_identical() {
        let v = vec![1.0, 2.0, 3.0];
        let d = Distance::Cosine.compute(&v, &v);
        assert!((d - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_distance_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let d = Distance::Cosine.compute(&a, &b);
        assert!((d - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_euclidean_distance() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![3.0, 4.0, 0.0];
        let d = Distance::Euclidean.compute(&a, &b);
        assert!((d - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_dot_product_distance() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let d = Distance::DotProduct.compute(&a, &b);
        // 1*4 + 2*5 + 3*6 = 32, negated = -32
        assert!((d - (-32.0)).abs() < 1e-6);
    }
}
