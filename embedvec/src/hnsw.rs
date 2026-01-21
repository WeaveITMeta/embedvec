//! HNSW (Hierarchical Navigable Small World) Index
//!
//! ## Table of Contents
//! - **HnswIndex**: Main HNSW graph structure for ANN search
//! - **HnswNode**: Node in the HNSW graph with multi-layer connections
//! - **HnswConfig**: Configuration parameters (M, ef_construction, ef_search)
//! - **Search Algorithm**: Greedy beam search with layer traversal
//!
//! ## Algorithm Overview
//! HNSW builds a multi-layer graph where:
//! - Higher layers have fewer nodes (exponential decay)
//! - Each node connects to M neighbors per layer
//! - Search starts from top layer, greedily descends
//! - Bottom layer search uses beam width ef_search for recall

use crate::distance::Distance;
use crate::e8::E8Codec;
use crate::error::Result;
use crate::storage::VectorStorage;

#[cfg(test)]
use crate::quantization::Quantization;
use ordered_float::OrderedFloat;
use rand::Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::BinaryHeap;

/// Fast bitset for tracking visited nodes during search
/// Much faster than HashSet for dense integer keys
struct VisitedSet {
    bits: Vec<u64>,
    generation: u64,
    generations: Vec<u64>,
}

impl VisitedSet {
    /// Create a new visited set with given capacity
    fn new(capacity: usize) -> Self {
        let num_words = (capacity + 63) / 64;
        Self {
            bits: vec![0; num_words],
            generation: 1,
            generations: vec![0; num_words],
        }
    }

    /// Clear the set (O(1) using generation trick)
    #[inline]
    #[allow(dead_code)]
    fn clear(&mut self) {
        self.generation += 1;
        // Handle overflow by resetting everything
        if self.generation == 0 {
            self.bits.fill(0);
            self.generations.fill(0);
            self.generation = 1;
        }
    }

    /// Check if id is visited, and mark it as visited
    /// Returns true if it was NOT visited (and is now marked)
    #[inline]
    fn insert(&mut self, id: usize) -> bool {
        let word_idx = id / 64;
        let bit_idx = id % 64;
        
        // Ensure capacity
        if word_idx >= self.bits.len() {
            let new_len = word_idx + 1;
            self.bits.resize(new_len, 0);
            self.generations.resize(new_len, 0);
        }
        
        // Check generation - if outdated, clear the word
        if self.generations[word_idx] != self.generation {
            self.bits[word_idx] = 0;
            self.generations[word_idx] = self.generation;
        }
        
        let mask = 1u64 << bit_idx;
        if self.bits[word_idx] & mask != 0 {
            false // Already visited
        } else {
            self.bits[word_idx] |= mask;
            true // Newly visited
        }
    }
}

/// HNSW node representing a vector in the graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnswNode {
    /// Vector ID in storage
    pub id: usize,
    /// Connections per layer (layer -> neighbor IDs)
    pub connections: Vec<Vec<usize>>,
    /// Maximum layer this node exists on
    pub max_layer: usize,
}

impl HnswNode {
    /// Create a new HNSW node
    pub fn new(id: usize, max_layer: usize, m: usize) -> Self {
        let connections = (0..=max_layer)
            .map(|_| Vec::with_capacity(m))
            .collect();
        Self {
            id,
            connections,
            max_layer,
        }
    }

    /// Get neighbors at a specific layer
    pub fn neighbors(&self, layer: usize) -> &[usize] {
        self.connections.get(layer).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Add a neighbor at a specific layer
    pub fn add_neighbor(&mut self, layer: usize, neighbor_id: usize) {
        if layer < self.connections.len() {
            if !self.connections[layer].contains(&neighbor_id) {
                self.connections[layer].push(neighbor_id);
            }
        }
    }
}

/// HNSW Index for approximate nearest neighbor search
#[derive(Debug)]
pub struct HnswIndex {
    /// All nodes in the graph
    nodes: Vec<HnswNode>,
    /// Entry point (node ID with highest layer)
    entry_point: Option<usize>,
    /// Maximum layer in the graph
    max_layer: usize,
    /// M: max connections per layer
    m: usize,
    /// M_max: max connections for layer 0
    m_max: usize,
    /// ef_construction: beam width during construction
    ef_construction: usize,
    /// Distance metric
    distance: Distance,
    /// Level multiplier for random layer assignment
    level_mult: f64,
}

impl HnswIndex {
    /// Create a new HNSW index
    ///
    /// # Arguments
    /// * `m` - Number of connections per layer (typically 16-64)
    /// * `ef_construction` - Beam width during construction (typically 100-500)
    /// * `distance` - Distance metric to use
    pub fn new(m: usize, ef_construction: usize, distance: Distance) -> Self {
        let m = m.max(2);
        let m_max = m * 2; // Layer 0 gets 2x connections
        let level_mult = 1.0 / (m as f64).ln();

        Self {
            nodes: Vec::new(),
            entry_point: None,
            max_layer: 0,
            m,
            m_max,
            ef_construction,
            distance,
            level_mult,
        }
    }

    /// Insert a vector into the index
    ///
    /// # Arguments
    /// * `id` - Vector ID in storage
    /// * `vector` - The vector data
    /// * `storage` - Vector storage for distance computation
    /// * `codec` - Optional E8 codec for quantized vectors
    pub fn insert(
        &mut self,
        id: usize,
        vector: &[f32],
        storage: &VectorStorage,
        codec: Option<&E8Codec>,
    ) -> Result<()> {
        // Assign random layer
        let node_layer = self.random_layer();

        // Create new node
        let mut new_node = HnswNode::new(id, node_layer, self.m);

        if self.entry_point.is_none() {
            // First node
            self.entry_point = Some(id);
            self.max_layer = node_layer;
            self.nodes.push(new_node);
            return Ok(());
        }

        let entry_point = self.entry_point.unwrap();

        // Search from top to node's layer
        let mut current_ep = entry_point;
        
        for layer in (node_layer + 1..=self.max_layer).rev() {
            current_ep = self.search_layer_single(vector, current_ep, layer, storage, codec)?;
        }

        // Search and connect at each layer from node_layer down to 0
        for layer in (0..=node_layer.min(self.max_layer)).rev() {
            let m_layer = if layer == 0 { self.m_max } else { self.m };

            // Find ef_construction nearest neighbors
            let neighbors = self.search_layer(
                vector,
                vec![current_ep],
                self.ef_construction,
                layer,
                storage,
                codec,
            )?;

            // Select M best neighbors
            let selected: Vec<usize> = neighbors
                .into_iter()
                .take(m_layer)
                .map(|(node_id, _)| node_id)
                .collect();

            // Connect new node to neighbors
            for &neighbor_id in &selected {
                new_node.add_neighbor(layer, neighbor_id);
            }

            // Connect neighbors back to new node (with pruning)
            // Collect nodes that need pruning first to avoid borrow conflicts
            let mut nodes_to_prune: Vec<usize> = Vec::new();
            
            for &neighbor_id in &selected {
                // Use direct indexing for O(1) lookup
                if let Some(neighbor) = self.nodes.get_mut(neighbor_id) {
                    if neighbor.connections.len() > layer {
                        neighbor.add_neighbor(layer, id);
                        
                        // Mark for pruning if too many connections
                        if neighbor.connections[layer].len() > m_layer {
                            nodes_to_prune.push(neighbor_id);
                        }
                    }
                }
            }
            
            // Now prune the marked nodes
            for neighbor_id in nodes_to_prune {
                self.prune_node_connections(neighbor_id, layer, m_layer, storage, codec)?;
            }

            // Update entry point for next layer
            if !selected.is_empty() {
                current_ep = selected[0];
            }
        }

        // Update entry point if new node has higher layer
        if node_layer > self.max_layer {
            self.entry_point = Some(id);
            self.max_layer = node_layer;
        }

        self.nodes.push(new_node);
        Ok(())
    }

    /// Search for k nearest neighbors
    ///
    /// # Arguments
    /// * `query` - Query vector
    /// * `k` - Number of neighbors to return
    /// * `ef_search` - Beam width for search (higher = better recall)
    /// * `storage` - Vector storage
    /// * `codec` - Optional E8 codec
    ///
    /// # Returns
    /// Vector of (id, distance) pairs sorted by distance
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        ef_search: usize,
        storage: &VectorStorage,
        codec: Option<&E8Codec>,
    ) -> Result<Vec<(usize, f32)>> {
        if self.entry_point.is_none() {
            return Ok(Vec::new());
        }

        let entry_point = self.entry_point.unwrap();
        let ef = ef_search.max(k);

        // Greedy search from top layer to layer 1
        let mut current_ep = entry_point;
        for layer in (1..=self.max_layer).rev() {
            current_ep = self.search_layer_single(query, current_ep, layer, storage, codec)?;
        }

        // Search layer 0 with full ef
        let candidates = self.search_layer(query, vec![current_ep], ef, 0, storage, codec)?;

        // Return top k
        Ok(candidates.into_iter().take(k).collect())
    }

    /// Search a single layer with beam search
    fn search_layer(
        &self,
        query: &[f32],
        entry_points: Vec<usize>,
        ef: usize,
        layer: usize,
        storage: &VectorStorage,
        codec: Option<&E8Codec>,
    ) -> Result<Vec<(usize, f32)>> {
        let mut visited = VisitedSet::new(self.nodes.len().max(1024));
        
        // Min-heap for candidates (closest first)
        let mut candidates: BinaryHeap<std::cmp::Reverse<(OrderedFloat<f32>, usize)>> =
            BinaryHeap::new();
        
        // Max-heap for results (furthest first, for pruning)
        let mut results: BinaryHeap<(OrderedFloat<f32>, usize)> = BinaryHeap::new();

        // Initialize with entry points
        for ep in entry_points {
            if visited.insert(ep) {
                let dist = self.compute_distance(query, ep, storage, codec)?;
                candidates.push(std::cmp::Reverse((OrderedFloat(dist), ep)));
                results.push((OrderedFloat(dist), ep));
            }
        }

        // Cache the furthest distance to avoid repeated heap peeks
        let mut furthest_dist = results.peek().map(|(d, _)| d.0).unwrap_or(f32::MAX);
        
        while let Some(std::cmp::Reverse((OrderedFloat(c_dist), c_id))) = candidates.pop() {
            // Early termination: if current candidate is worse than worst result
            if c_dist > furthest_dist && results.len() >= ef {
                break;
            }

            // Explore neighbors - use direct indexing for O(1) lookup
            if let Some(node) = self.nodes.get(c_id) {
                for &neighbor_id in node.neighbors(layer) {
                    if visited.insert(neighbor_id) {
                        let dist = self.compute_distance(query, neighbor_id, storage, codec)?;

                        if results.len() < ef || dist < furthest_dist {
                            candidates.push(std::cmp::Reverse((OrderedFloat(dist), neighbor_id)));
                            results.push((OrderedFloat(dist), neighbor_id));

                            if results.len() > ef {
                                results.pop();
                                // Update cached furthest distance after pop
                                furthest_dist = results.peek().map(|(d, _)| d.0).unwrap_or(f32::MAX);
                            }
                        }
                    }
                }
            }
        }

        // Convert results to sorted vector
        let mut result_vec: Vec<(usize, f32)> = results
            .into_iter()
            .map(|(d, id)| (id, d.0))
            .collect();
        result_vec.sort_by_key(|(_, d)| OrderedFloat(*d));

        Ok(result_vec)
    }

    /// Search layer returning single best entry point
    fn search_layer_single(
        &self,
        query: &[f32],
        entry_point: usize,
        layer: usize,
        storage: &VectorStorage,
        codec: Option<&E8Codec>,
    ) -> Result<usize> {
        let mut current = entry_point;
        let mut current_dist = self.compute_distance(query, current, storage, codec)?;

        loop {
            let mut changed = false;

            // Use direct indexing for O(1) lookup
            if let Some(node) = self.nodes.get(current) {
                for &neighbor_id in node.neighbors(layer) {
                    let dist = self.compute_distance(query, neighbor_id, storage, codec)?;
                    if dist < current_dist {
                        current = neighbor_id;
                        current_dist = dist;
                        changed = true;
                    }
                }
            }

            if !changed {
                break;
            }
        }

        Ok(current)
    }

    /// Compute distance between query and stored vector
    /// Uses zero-copy access for raw vectors when possible
    #[inline]
    fn compute_distance(
        &self,
        query: &[f32],
        id: usize,
        storage: &VectorStorage,
        codec: Option<&E8Codec>,
    ) -> Result<f32> {
        // Try zero-copy path for raw vectors first
        if let Some(raw_slice) = storage.get_raw_slice(id) {
            return Ok(self.distance.compute(query, raw_slice));
        }
        // Fall back to decoding for quantized vectors
        let vector = storage.get(id, codec)?;
        Ok(self.distance.compute(query, &vector))
    }

    /// Prune connections for a node by ID to keep only M best
    fn prune_node_connections(
        &mut self,
        node_id: usize,
        layer: usize,
        m: usize,
        storage: &VectorStorage,
        codec: Option<&E8Codec>,
    ) -> Result<()> {
        // Get node vector and current connections
        let node_vector = storage.get(node_id, codec)?;
        
        let connections = {
            // Use direct indexing for O(1) lookup
            match self.nodes.get(node_id) {
                Some(n) if n.connections.len() > layer => n.connections[layer].clone(),
                _ => return Ok(()),
            }
        };
        
        // Compute distances to all neighbors
        let mut neighbor_dists: Vec<(usize, f32)> = connections
            .iter()
            .filter_map(|&neighbor_id| {
                storage.get(neighbor_id, codec).ok().map(|v| {
                    let dist = self.distance.compute(&node_vector, &v);
                    (neighbor_id, dist)
                })
            })
            .collect();

        // Sort by distance and keep best M
        neighbor_dists.sort_by_key(|(_, d)| OrderedFloat(*d));
        let pruned: Vec<usize> = neighbor_dists.into_iter().take(m).map(|(id, _)| id).collect();
        
        // Update the node's connections - use direct indexing
        if let Some(node) = self.nodes.get_mut(node_id) {
            if node.connections.len() > layer {
                node.connections[layer] = pruned;
            }
        }

        Ok(())
    }

    /// Generate random layer for new node
    fn random_layer(&self) -> usize {
        let mut rng = rand::thread_rng();
        let r: f64 = rng.gen();
        (-r.ln() * self.level_mult).floor() as usize
    }

    /// Clear the index
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.entry_point = None;
        self.max_layer = 0;
    }

    /// Get number of nodes in index
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Check if index is empty
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Get index statistics
    pub fn stats(&self) -> HnswStats {
        let total_connections: usize = self.nodes
            .iter()
            .map(|n| n.connections.iter().map(|c| c.len()).sum::<usize>())
            .sum();

        let avg_connections = if self.nodes.is_empty() {
            0.0
        } else {
            total_connections as f64 / self.nodes.len() as f64
        };

        HnswStats {
            num_nodes: self.nodes.len(),
            max_layer: self.max_layer,
            total_connections,
            avg_connections_per_node: avg_connections,
            m: self.m,
            ef_construction: self.ef_construction,
        }
    }

    /// Parallel batch search for multiple queries
    /// Uses rayon for parallel execution across queries
    ///
    /// # Arguments
    /// * `queries` - Slice of query vectors
    /// * `k` - Number of neighbors per query
    /// * `ef_search` - Beam width for search
    /// * `storage` - Vector storage
    /// * `codec` - Optional E8 codec
    ///
    /// # Returns
    /// Vector of results, one per query
    pub fn search_batch(
        &self,
        queries: &[Vec<f32>],
        k: usize,
        ef_search: usize,
        storage: &VectorStorage,
        codec: Option<&E8Codec>,
    ) -> Vec<Result<Vec<(usize, f32)>>> {
        queries
            .par_iter()
            .map(|query| self.search(query, k, ef_search, storage, codec))
            .collect()
    }
}

/// HNSW index statistics
#[derive(Debug, Clone)]
pub struct HnswStats {
    /// Number of nodes in the index
    pub num_nodes: usize,
    /// Maximum layer height
    pub max_layer: usize,
    /// Total number of connections
    pub total_connections: usize,
    /// Average connections per node
    pub avg_connections_per_node: f64,
    /// M parameter
    pub m: usize,
    /// ef_construction parameter
    pub ef_construction: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    fn create_test_storage(vectors: &[Vec<f32>]) -> VectorStorage {
        let dim = vectors.first().map(|v| v.len()).unwrap_or(4);
        let mut storage = VectorStorage::new(dim, Quantization::None);
        for v in vectors {
            storage.add(v, None).unwrap();
        }
        storage
    }

    #[test]
    fn test_hnsw_insert_single() {
        let mut index = HnswIndex::new(16, 100, Distance::Euclidean);
        let storage = create_test_storage(&[vec![1.0, 0.0, 0.0, 0.0]]);

        index.insert(0, &[1.0, 0.0, 0.0, 0.0], &storage, None).unwrap();
        assert_eq!(index.len(), 1);
    }

    #[test]
    fn test_hnsw_search_basic() {
        let mut index = HnswIndex::new(16, 100, Distance::Euclidean);
        
        let vectors = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
        ];
        let storage = create_test_storage(&vectors);

        for (id, v) in vectors.iter().enumerate() {
            index.insert(id, v, &storage, None).unwrap();
        }

        // Search for vector close to first
        let query = vec![0.9, 0.1, 0.0, 0.0];
        let results = index.search(&query, 2, 50, &storage, None).unwrap();

        assert!(!results.is_empty());
        assert_eq!(results[0].0, 0); // Closest should be first vector
    }

    #[test]
    fn test_hnsw_search_empty() {
        let index = HnswIndex::new(16, 100, Distance::Euclidean);
        let storage = VectorStorage::new(4, Quantization::None);

        let results = index.search(&[1.0, 0.0, 0.0, 0.0], 5, 50, &storage, None).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_hnsw_stats() {
        let mut index = HnswIndex::new(4, 50, Distance::Euclidean);
        
        let vectors: Vec<Vec<f32>> = (0..10)
            .map(|i| vec![i as f32, 0.0, 0.0, 0.0])
            .collect();
        let storage = create_test_storage(&vectors);

        for (id, v) in vectors.iter().enumerate() {
            index.insert(id, v, &storage, None).unwrap();
        }

        let stats = index.stats();
        assert_eq!(stats.num_nodes, 10);
        assert!(stats.avg_connections_per_node > 0.0);
    }

    #[test]
    fn test_hnsw_clear() {
        let mut index = HnswIndex::new(16, 100, Distance::Euclidean);
        let storage = create_test_storage(&[vec![1.0, 0.0, 0.0, 0.0]]);

        index.insert(0, &[1.0, 0.0, 0.0, 0.0], &storage, None).unwrap();
        assert_eq!(index.len(), 1);

        index.clear();
        assert_eq!(index.len(), 0);
        assert!(index.is_empty());
    }
}
