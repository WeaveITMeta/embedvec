//! Vector Storage Module
//!
//! ## Table of Contents
//! - **VectorStorage**: Main storage for vectors (raw f32, E8-quantized, or H4-quantized)
//! - **StoredVector**: Enum representing stored vector format
//! - **Storage operations**: add, get, clear, re-quantization

use crate::e8::{E8Codec, E8EncodedVector};
use crate::h4::{H4Codec, H4EncodedVector};
use crate::error::{EmbedVecError, Result};
use crate::quantization::Quantization;
use serde::{Deserialize, Serialize};

/// Stored vector representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StoredVector {
    /// Raw f32 vector (no quantization)
    Raw(Vec<f32>),
    /// E8-quantized vector (8D blocks, D8 ∪ D8+½ decomposition)
    E8(E8EncodedVector),
    /// H4-quantized vector (4D blocks, 600-cell vertex indices)
    H4(H4EncodedVector),
    /// Tombstone placeholder for a deleted slot.
    ///
    /// Holds no data and is never part of the HNSW graph, so it is never
    /// traversed or returned by a search. It exists only to keep vector IDs
    /// stable (id == position) across a delete + reopen cycle.
    Tombstone,
}

impl StoredVector {
    /// Get the raw f32 vector (decoding if necessary)
    pub fn to_f32(&self, e8_codec: Option<&E8Codec>, h4_codec: Option<&H4Codec>) -> Vec<f32> {
        match self {
            StoredVector::Raw(v) => v.clone(),
            StoredVector::E8(encoded) => {
                if let Some(c) = e8_codec {
                    c.decode(encoded)
                } else {
                    // Fallback: return zeros if no codec (shouldn't happen)
                    vec![0.0; encoded.points.len() * 8]
                }
            }
            StoredVector::H4(encoded) => {
                if let Some(c) = h4_codec {
                    c.decode(encoded)
                } else {
                    // Fallback: return zeros if no codec (shouldn't happen)
                    vec![0.0; encoded.indices.len() * 4]
                }
            }
            // Deleted slot: no data to decode.
            StoredVector::Tombstone => Vec::new(),
        }
    }

    /// Get size in bytes
    pub fn size_bytes(&self) -> usize {
        match self {
            StoredVector::Raw(v) => v.len() * 4,
            StoredVector::E8(encoded) => encoded.size_bytes(),
            StoredVector::H4(encoded) => encoded.size_bytes(),
            StoredVector::Tombstone => 0,
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
    /// * `e8_codec` - Optional E8 codec (required when quantization is E8)
    /// * `h4_codec` - Optional H4 codec (required when quantization is H4)
    ///
    /// # Returns
    /// Assigned vector ID
    pub fn add(
        &mut self,
        vector: &[f32],
        e8_codec: Option<&E8Codec>,
        h4_codec: Option<&H4Codec>,
    ) -> Result<usize> {
        let stored = match &self.quantization {
            Quantization::None => StoredVector::Raw(vector.to_vec()),
            Quantization::E8 { .. } => {
                if let Some(c) = e8_codec {
                    let encoded = c.encode(vector)?;
                    StoredVector::E8(encoded)
                } else {
                    return Err(EmbedVecError::QuantizationError(
                        "E8 codec required for E8 quantization".to_string(),
                    ));
                }
            }
            Quantization::H4 { .. } => {
                if let Some(c) = h4_codec {
                    let encoded = c.encode(vector)?;
                    StoredVector::H4(encoded)
                } else {
                    return Err(EmbedVecError::QuantizationError(
                        "H4 codec required for H4 quantization".to_string(),
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
    /// * `e8_codec` - Optional E8 codec for decoding
    /// * `h4_codec` - Optional H4 codec for decoding
    ///
    /// # Returns
    /// Raw f32 vector (decoded if quantized)
    #[inline]
    pub fn get(
        &self,
        id: usize,
        e8_codec: Option<&E8Codec>,
        h4_codec: Option<&H4Codec>,
    ) -> Result<Vec<f32>> {
        self.vectors
            .get(id)
            .map(|v| v.to_f32(e8_codec, h4_codec))
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

    /// Append a pre-encoded stored vector without re-encoding.
    ///
    /// Used when reloading persisted vectors from disk: the on-disk form is
    /// already a `StoredVector`, so we push it directly and only need to track
    /// memory. Returns the assigned vector ID.
    pub fn push_stored(&mut self, stored: StoredVector) -> usize {
        self.memory_bytes += stored.size_bytes();
        let id = self.vectors.len();
        self.vectors.push(stored);
        id
    }
    
    /// Batch get multiple vectors by IDs (more efficient than individual gets)
    pub fn get_batch(
        &self,
        ids: &[usize],
        e8_codec: Option<&E8Codec>,
        h4_codec: Option<&H4Codec>,
    ) -> Vec<Option<Vec<f32>>> {
        ids.iter()
            .map(|&id| self.vectors.get(id).map(|v| v.to_f32(e8_codec, h4_codec)))
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
    /// * `e8_codec` - E8 codec (required if switching to or from E8)
    /// * `h4_codec` - H4 codec (required if switching to or from H4)
    pub fn set_quantization(
        &mut self,
        new_quantization: Quantization,
        e8_codec: Option<&E8Codec>,
        h4_codec: Option<&H4Codec>,
    ) -> Result<()> {
        if self.quantization == new_quantization {
            return Ok(());
        }

        // Re-quantize all vectors
        let mut new_vectors = Vec::with_capacity(self.vectors.len());
        let mut new_memory = 0usize;

        for stored in &self.vectors {
            // Deleted slots carry no data — keep them as tombstones.
            if matches!(stored, StoredVector::Tombstone) {
                new_vectors.push(StoredVector::Tombstone);
                continue;
            }

            // First decode to f32 using whichever codec applies to current format
            let raw = stored.to_f32(e8_codec, h4_codec);

            // Then encode with new quantization
            let new_stored = match &new_quantization {
                Quantization::None => StoredVector::Raw(raw),
                Quantization::E8 { .. } => {
                    if let Some(c) = e8_codec {
                        let encoded = c.encode(&raw)?;
                        StoredVector::E8(encoded)
                    } else {
                        return Err(EmbedVecError::QuantizationError(
                            "E8 codec required for E8 quantization".to_string(),
                        ));
                    }
                }
                Quantization::H4 { .. } => {
                    if let Some(c) = h4_codec {
                        let encoded = c.encode(&raw)?;
                        StoredVector::H4(encoded)
                    } else {
                        return Err(EmbedVecError::QuantizationError(
                            "H4 codec required for H4 quantization".to_string(),
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
        e8_codec: Option<&E8Codec>,
        h4_codec: Option<&H4Codec>,
        distance_fn: impl Fn(&[f32], &[f32]) -> f32,
    ) -> Result<f32> {
        let stored = self
            .vectors
            .get(id)
            .ok_or(EmbedVecError::VectorNotFound(id))?;

        match stored {
            StoredVector::Raw(v) => Ok(distance_fn(query, v)),
            StoredVector::E8(encoded) => {
                if let Some(c) = e8_codec {
                    let decoded = c.decode(encoded);
                    Ok(distance_fn(query, &decoded))
                } else {
                    Err(EmbedVecError::QuantizationError(
                        "E8 codec required for distance computation".to_string(),
                    ))
                }
            }
            StoredVector::H4(encoded) => {
                if let Some(c) = h4_codec {
                    let decoded = c.decode(encoded);
                    Ok(distance_fn(query, &decoded))
                } else {
                    Err(EmbedVecError::QuantizationError(
                        "H4 codec required for distance computation".to_string(),
                    ))
                }
            }
            // Deleted slots are never in the graph; treat as infinitely far if reached.
            StoredVector::Tombstone => Ok(f32::INFINITY),
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
        let id = storage.add(&v1, None, None).unwrap();
        assert_eq!(id, 0);

        let retrieved = storage.get(0, None, None).unwrap();
        assert_eq!(retrieved, v1);
    }

    #[test]
    fn test_e8_storage() {
        use crate::e8::E8Codec;
        let codec = E8Codec::new(16, 10, true, 42);
        let mut storage = VectorStorage::new(16, Quantization::e8_default());

        let v1: Vec<f32> = (0..16).map(|i| i as f32 * 0.1).collect();
        let id = storage.add(&v1, Some(&codec), None).unwrap();
        assert_eq!(id, 0);

        let retrieved = storage.get(0, Some(&codec), None).unwrap();
        assert_eq!(retrieved.len(), 16);

        // Check that it's approximately equal (quantization introduces error)
        let mse: f32 = v1
            .iter()
            .zip(retrieved.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / 16.0;
        assert!(mse < 1.0, "MSE too high: {}", mse);
    }

    #[test]
    fn test_h4_storage() {
        use crate::h4::H4Codec;
        let codec = H4Codec::new(16, true, 42);
        let mut storage = VectorStorage::new(16, Quantization::h4_default());

        let v1: Vec<f32> = (0..16).map(|i| (i as f32 * 0.3).sin()).collect();
        let id = storage.add(&v1, None, Some(&codec)).unwrap();
        assert_eq!(id, 0);

        let retrieved = storage.get(0, None, Some(&codec)).unwrap();
        assert_eq!(retrieved.len(), 16);

        // H4 is a lossy quantizer
        let mse: f32 = v1
            .iter()
            .zip(retrieved.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / 16.0;
        assert!(mse < 1.0, "H4 MSE too high: {}", mse);
    }

    #[test]
    fn test_memory_tracking() {
        let mut storage = VectorStorage::new(768, Quantization::None);

        for _ in 0..10 {
            let v: Vec<f32> = vec![0.0; 768];
            storage.add(&v, None, None).unwrap();
        }

        assert_eq!(storage.memory_bytes(), 768 * 4 * 10);
    }

    #[test]
    fn test_clear() {
        let mut storage = VectorStorage::new(4, Quantization::None);
        storage.add(&[1.0, 2.0, 3.0, 4.0], None, None).unwrap();
        
        assert_eq!(storage.len(), 1);
        storage.clear();
        assert_eq!(storage.len(), 0);
        assert_eq!(storage.memory_bytes(), 0);
    }
}
