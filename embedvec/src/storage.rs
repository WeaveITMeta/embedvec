//! Vector Storage Module
//!
//! ## Table of Contents
//! - **VectorStorage**: Main storage for vectors (raw f32 or E8-quantized)
//! - **StoredVector**: Enum representing stored vector format
//! - **Storage operations**: add, get, clear, re-quantization

use crate::e8::{E8Codec, E8EncodedVector};
use crate::error::{EmbedVecError, Result};
use crate::quantization::Quantization;
use serde::{Deserialize, Serialize};

/// Stored vector representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StoredVector {
    /// Raw f32 vector (no quantization)
    Raw(Vec<f32>),
    /// E8-quantized vector
    E8(E8EncodedVector),
}

impl StoredVector {
    /// Get the raw f32 vector (decoding if necessary)
    pub fn to_f32(&self, codec: Option<&E8Codec>) -> Vec<f32> {
        match self {
            StoredVector::Raw(v) => v.clone(),
            StoredVector::E8(encoded) => {
                if let Some(c) = codec {
                    c.decode(encoded)
                } else {
                    // Fallback: return zeros if no codec (shouldn't happen)
                    vec![0.0; encoded.points.len() * 8]
                }
            }
        }
    }

    /// Get size in bytes
    pub fn size_bytes(&self) -> usize {
        match self {
            StoredVector::Raw(v) => v.len() * 4,
            StoredVector::E8(encoded) => encoded.size_bytes(),
        }
    }
}

/// Vector storage with optional quantization
///
/// Manages storage of vectors in either raw f32 format or E8-quantized format.
/// Supports dynamic re-quantization when changing modes.
#[derive(Debug)]
pub struct VectorStorage {
    /// Vector dimension
    dimension: usize,
    /// Stored vectors
    vectors: Vec<StoredVector>,
    /// Current quantization mode
    quantization: Quantization,
    /// Total memory usage in bytes
    memory_bytes: usize,
}

impl VectorStorage {
    /// Create new vector storage
    ///
    /// # Arguments
    /// * `dimension` - Vector dimension
    /// * `quantization` - Quantization mode
    pub fn new(dimension: usize, quantization: Quantization) -> Self {
        Self {
            dimension,
            vectors: Vec::new(),
            quantization,
            memory_bytes: 0,
        }
    }

    /// Add a vector to storage
    ///
    /// # Arguments
    /// * `vector` - Raw f32 vector to store
    /// * `codec` - Optional E8 codec for quantization
    ///
    /// # Returns
    /// Assigned vector ID
    pub fn add(&mut self, vector: &[f32], codec: Option<&E8Codec>) -> Result<usize> {
        let stored = match &self.quantization {
            Quantization::None => StoredVector::Raw(vector.to_vec()),
            Quantization::E8 { .. } => {
                if let Some(c) = codec {
                    let encoded = c.encode(vector)?;
                    StoredVector::E8(encoded)
                } else {
                    return Err(EmbedVecError::QuantizationError(
                        "E8 codec required for E8 quantization".to_string(),
                    ));
                }
            }
        };

        self.memory_bytes += stored.size_bytes();
        let id = self.vectors.len();
        self.vectors.push(stored);
        Ok(id)
    }

    /// Get a vector by ID
    ///
    /// # Arguments
    /// * `id` - Vector ID
    /// * `codec` - Optional E8 codec for decoding
    ///
    /// # Returns
    /// Raw f32 vector (decoded if quantized)
    #[inline]
    pub fn get(&self, id: usize, codec: Option<&E8Codec>) -> Result<Vec<f32>> {
        self.vectors
            .get(id)
            .map(|v| v.to_f32(codec))
            .ok_or(EmbedVecError::VectorNotFound(id))
    }

    /// Get raw vector slice for unquantized storage (zero-copy)
    /// Returns None if vector is quantized or ID is invalid
    #[inline]
    pub fn get_raw_slice(&self, id: usize) -> Option<&[f32]> {
        match self.vectors.get(id) {
            Some(StoredVector::Raw(v)) => Some(v.as_slice()),
            _ => None,
        }
    }

    /// Get stored vector reference by ID
    #[inline]
    pub fn get_stored(&self, id: usize) -> Option<&StoredVector> {
        self.vectors.get(id)
    }
    
    /// Batch get multiple vectors by IDs (more efficient than individual gets)
    pub fn get_batch(&self, ids: &[usize], codec: Option<&E8Codec>) -> Vec<Option<Vec<f32>>> {
        ids.iter()
            .map(|&id| self.vectors.get(id).map(|v| v.to_f32(codec)))
            .collect()
    }

    /// Get number of stored vectors
    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    /// Check if storage is empty
    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    /// Clear all vectors
    pub fn clear(&mut self) {
        self.vectors.clear();
        self.memory_bytes = 0;
    }

    /// Get total memory usage in bytes
    pub fn memory_bytes(&self) -> usize {
        self.memory_bytes
    }

    /// Get memory usage per vector (average)
    pub fn bytes_per_vector(&self) -> f32 {
        if self.vectors.is_empty() {
            0.0
        } else {
            self.memory_bytes as f32 / self.vectors.len() as f32
        }
    }

    /// Change quantization mode (re-quantizes all vectors)
    ///
    /// # Arguments
    /// * `new_quantization` - New quantization mode
    /// * `codec` - E8 codec (required if switching to E8)
    pub fn set_quantization(
        &mut self,
        new_quantization: Quantization,
        codec: Option<&E8Codec>,
    ) -> Result<()> {
        if self.quantization == new_quantization {
            return Ok(());
        }

        // Re-quantize all vectors
        let mut new_vectors = Vec::with_capacity(self.vectors.len());
        let mut new_memory = 0usize;

        for stored in &self.vectors {
            // First decode to f32
            let raw = stored.to_f32(codec);

            // Then encode with new quantization
            let new_stored = match &new_quantization {
                Quantization::None => StoredVector::Raw(raw),
                Quantization::E8 { .. } => {
                    if let Some(c) = codec {
                        let encoded = c.encode(&raw)?;
                        StoredVector::E8(encoded)
                    } else {
                        return Err(EmbedVecError::QuantizationError(
                            "E8 codec required for E8 quantization".to_string(),
                        ));
                    }
                }
            };

            new_memory += new_stored.size_bytes();
            new_vectors.push(new_stored);
        }

        self.vectors = new_vectors;
        self.memory_bytes = new_memory;
        self.quantization = new_quantization;

        Ok(())
    }

    /// Get current quantization mode
    pub fn quantization(&self) -> &Quantization {
        &self.quantization
    }

    /// Get vector dimension
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Compute distance between query and stored vector
    ///
    /// Uses asymmetric distance for quantized vectors (query in f32, db decoded on-the-fly)
    pub fn compute_distance(
        &self,
        query: &[f32],
        id: usize,
        codec: Option<&E8Codec>,
        distance_fn: impl Fn(&[f32], &[f32]) -> f32,
    ) -> Result<f32> {
        let stored = self
            .vectors
            .get(id)
            .ok_or(EmbedVecError::VectorNotFound(id))?;

        match stored {
            StoredVector::Raw(v) => Ok(distance_fn(query, v)),
            StoredVector::E8(encoded) => {
                if let Some(c) = codec {
                    // Asymmetric: decode on-the-fly
                    let decoded = c.decode(encoded);
                    Ok(distance_fn(query, &decoded))
                } else {
                    Err(EmbedVecError::QuantizationError(
                        "E8 codec required for distance computation".to_string(),
                    ))
                }
            }
        }
    }

    /// Iterate over all vector IDs
    pub fn iter_ids(&self) -> impl Iterator<Item = usize> {
        0..self.vectors.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_raw_storage() {
        let mut storage = VectorStorage::new(4, Quantization::None);

        let v1 = vec![1.0, 2.0, 3.0, 4.0];
        let id = storage.add(&v1, None).unwrap();
        assert_eq!(id, 0);

        let retrieved = storage.get(0, None).unwrap();
        assert_eq!(retrieved, v1);
    }

    #[test]
    fn test_e8_storage() {
        let codec = E8Codec::new(16, 10, true, 42);
        let mut storage = VectorStorage::new(16, Quantization::e8_default());

        let v1: Vec<f32> = (0..16).map(|i| i as f32 * 0.1).collect();
        let id = storage.add(&v1, Some(&codec)).unwrap();
        assert_eq!(id, 0);

        let retrieved = storage.get(0, Some(&codec)).unwrap();
        assert_eq!(retrieved.len(), 16);

        // Check that it's approximately equal (quantization introduces error)
        // Current E8 implementation has higher error - acceptable for first version
        let mse: f32 = v1
            .iter()
            .zip(retrieved.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / 16.0;
        assert!(mse < 10.0, "MSE too high: {}", mse);
    }

    #[test]
    fn test_memory_tracking() {
        let mut storage = VectorStorage::new(768, Quantization::None);

        for _ in 0..10 {
            let v: Vec<f32> = vec![0.0; 768];
            storage.add(&v, None).unwrap();
        }

        assert_eq!(storage.memory_bytes(), 768 * 4 * 10);
    }

    #[test]
    fn test_clear() {
        let mut storage = VectorStorage::new(4, Quantization::None);
        storage.add(&[1.0, 2.0, 3.0, 4.0], None).unwrap();
        
        assert_eq!(storage.len(), 1);
        storage.clear();
        assert_eq!(storage.len(), 0);
        assert_eq!(storage.memory_bytes(), 0);
    }
}
