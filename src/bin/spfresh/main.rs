mod datasets;

use clap::Parser;
use datasets::{Dataset, DatasetType};
use rand::Rng;
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read as IoRead, Write};
use std::path::Path;
use std::time::Instant;
use usearch::{new_index, Index, IndexOptions, MetricKind, ScalarKind};

/// SPFresh-based vector index configuration
#[derive(Clone)]
pub struct SPFreshConfig {
    /// Vector dimensions (fixed for all vectors)
    pub dimensions: usize,
    /// Maximum vectors per posting list before split (N)
    pub max_posting_size: usize,
    /// Minimum vectors per posting list before merge (M)
    pub min_posting_size: usize,
    /// Number of centroids to probe during search
    pub num_probes: usize,
    /// Number of neighbors to check for reassignment after split
    pub reassign_neighbors: usize,
    /// K-means iterations for split
    pub kmeans_iters: usize,
    /// HNSW connectivity (M parameter) - number of edges per node
    pub connectivity: usize,
    /// HNSW expansion factor during index construction
    pub expansion_add: usize,
    /// HNSW expansion factor during search
    pub expansion_search: usize,
    /// Query-aware dynamic pruning threshold (epsilon from SPANN paper)
    /// Only search posting lists where: dist(q, centroid) <= (1 + epsilon) * dist(q, closest_centroid)
    /// Set to 0.0 to disable pruning (search all num_probes posting lists)
    pub pruning_threshold: f32,
    /// Multi-cluster assignment threshold (epsilon1 from SPANN paper)
    /// Assign vector to multiple clusters where: dist(v, c) <= (1 + epsilon1) * dist(v, closest_c)
    /// Set to 0.0 to disable (single assignment). Typical values: 0.1-0.5
    pub assignment_threshold: f32,
    /// Maximum number of centroids a vector can be assigned to (caps replication)
    /// Set to 0 for unlimited. Typical values: 4-16
    pub max_assignment_count: usize,
}

impl Default for SPFreshConfig {
    fn default() -> Self {
        Self {
            dimensions: 128,
            max_posting_size: 1000,
            min_posting_size: 100,
            num_probes: 10,
            reassign_neighbors: 5,
            kmeans_iters: 10,
            connectivity: 16,
            expansion_add: 128,
            expansion_search: 64,
            pruning_threshold: 0.0, // disabled by default
            assignment_threshold: 0.0, // disabled by default (single assignment)
            max_assignment_count: 0, // 0 = unlimited
        }
    }
}

/// A vector stored in a posting list
#[derive(Clone)]
struct StoredVector {
    id: u64,
    data: Vec<f32>,
}

/// SPFresh-based approximate nearest neighbor index
pub struct SPFreshIndex {
    /// Configuration
    config: SPFreshConfig,
    /// Centroid index (usearch HNSW)
    centroid_index: Index,
    /// Posting lists: centroid_id -> vectors assigned to this centroid
    posting_lists: HashMap<u64, Vec<StoredVector>>,
    /// Centroid vectors (needed for distance calculations and k-means)
    centroid_vectors: HashMap<u64, Vec<f32>>,
    /// Vector assignments: vector_id -> set of centroid_ids it belongs to
    /// Used for multi-cluster assignment (SPANN posting list expansion)
    vector_assignments: HashMap<u64, HashSet<u64>>,
    /// Vector data: vector_id -> vector data (canonical storage for deduplication)
    vector_data: HashMap<u64, Vec<f32>>,
    /// Next centroid ID
    next_centroid_id: u64,
    /// Total vectors in the index (unique vectors, not posting list entries)
    total_vectors: usize,
    /// Number of splits performed
    num_splits: usize,
    /// Number of merges performed
    num_merges: usize,
    /// Number of vectors reassigned
    num_reassigned: usize,
}

impl SPFreshIndex {
    /// Create a new SPFresh index with initial zero centroid
    pub fn new(config: SPFreshConfig) -> Self {
        let index_options = IndexOptions {
            dimensions: config.dimensions,
            metric: MetricKind::L2sq,
            quantization: ScalarKind::F32,
            connectivity: config.connectivity,
            expansion_add: config.expansion_add,
            expansion_search: config.expansion_search,
            multi: false,
        };

        let centroid_index = new_index(&index_options).expect("Failed to create centroid index");
        centroid_index.reserve(100000).expect("Failed to reserve");

        // Initialize with a single zero centroid
        let zero_centroid: Vec<f32> = vec![0.0; config.dimensions];
        centroid_index
            .add(0, &zero_centroid)
            .expect("Failed to add initial centroid");

        let mut centroid_vectors = HashMap::new();
        centroid_vectors.insert(0, zero_centroid);

        let mut posting_lists = HashMap::new();
        posting_lists.insert(0, Vec::new());

        Self {
            config,
            centroid_index,
            posting_lists,
            centroid_vectors,
            vector_assignments: HashMap::new(),
            vector_data: HashMap::new(),
            next_centroid_id: 1,
            total_vectors: 0,
            num_splits: 0,
            num_merges: 0,
            num_reassigned: 0,
        }
    }

    /// Insert a vector into the index
    pub fn insert(&mut self, id: u64, vector: &[f32]) {
        assert_eq!(
            vector.len(),
            self.config.dimensions,
            "Vector dimension mismatch"
        );

        // Compute multi-cluster assignments (or single assignment if threshold is 0)
        let assignments = self.compute_multi_assignments(vector);

        // Add to posting lists and tracking structures
        self.add_vector_to_assignments(id, vector.to_vec(), assignments.clone());
        self.total_vectors += 1;

        // Check if any assigned centroid needs splitting
        for centroid_id in assignments {
            if self.posting_lists.get(&centroid_id).map(|l| l.len()).unwrap_or(0)
                > self.config.max_posting_size
            {
                self.split(centroid_id);
            }
        }
    }

    fn add_to_centroids_index(&mut self, id: u64, v: &[f32]) {
        if let Err(_e) = self.centroid_index.add(id, v) {
            println!("RESERVING: {}", self.centroid_vectors.len() + 1024);
            self.centroid_index.reserve(self.centroid_vectors.len() + 1024).expect("Failed to reserve");
            self.centroid_index.add(id, v).expect("Failed to add to index");
        }
        self.centroid_vectors.insert(id, v.into());
    }

    /// Compute multi-cluster assignments for a vector using SPANN posting list expansion.
    /// Returns the set of centroid IDs the vector should be assigned to.
    ///
    /// Uses two rules from the SPANN paper:
    /// 1. Distance threshold: assign to cluster c if dist(v, c) <= (1 + ε) * dist(v, closest_c)
    /// 2. RNG rule: skip cluster c_j if dist(c_j, v) > dist(c_{j-1}, c_j) to avoid redundant
    ///    assignments to nearby clusters that would likely be searched together
    fn compute_multi_assignments(&self, vector: &[f32]) -> HashSet<u64> {
        let threshold = self.config.assignment_threshold;

        // If threshold is 0, just return single nearest centroid
        if threshold <= 0.0 {
            let nearest = self.centroid_index.search(vector, 1).expect("Search failed");
            let mut result = HashSet::new();
            result.insert(nearest.keys[0]);
            return result;
        }

        // Search for more centroids than we'll likely assign (to have candidates for RNG filtering)
        let num_candidates = (self.config.num_probes * 2).min(self.centroid_vectors.len());
        let search_results = self.centroid_index.search(vector, num_candidates).expect("Search failed");

        if search_results.keys.is_empty() {
            return HashSet::new();
        }

        let min_dist = search_results.distances[0];
        let dist_threshold = (1.0 + threshold) * min_dist;

        // Collect candidates that pass the distance threshold
        let mut candidates: Vec<(u64, f32)> = Vec::new();
        for (i, &centroid_id) in search_results.keys.iter().enumerate() {
            let dist = search_results.distances[i];
            if dist <= dist_threshold {
                candidates.push((centroid_id, dist));
            }
        }

        // Simple multi-cluster assignment: include all centroids within the distance threshold
        // Optionally limited by max_assignment_count
        let mut result = HashSet::new();
        let max_count = if self.config.max_assignment_count > 0 {
            self.config.max_assignment_count
        } else {
            usize::MAX
        };

        for (centroid_id, _) in &candidates {
            if result.len() >= max_count {
                break;
            }
            result.insert(*centroid_id);
        }

        result
    }

    /// Remove a vector from all its current posting lists.
    /// Returns the vector data if found.
    fn remove_vector_from_all_postings(&mut self, vector_id: u64) -> Option<Vec<f32>> {
        // Get current assignments
        let assignments = match self.vector_assignments.remove(&vector_id) {
            Some(a) => a,
            None => return self.vector_data.remove(&vector_id),
        };

        // Remove from all posting lists
        for centroid_id in assignments {
            if let Some(posting_list) = self.posting_lists.get_mut(&centroid_id) {
                if let Some(pos) = posting_list.iter().position(|v| v.id == vector_id) {
                    posting_list.swap_remove(pos);
                }
            }
        }

        // Return and remove from vector_data
        self.vector_data.remove(&vector_id)
    }

    /// Add a vector to posting lists based on computed assignments.
    /// Updates vector_assignments and vector_data.
    fn add_vector_to_assignments(&mut self, vector_id: u64, vector: Vec<f32>, assignments: HashSet<u64>) {
        // Store canonical vector data
        self.vector_data.insert(vector_id, vector.clone());

        // Add to each assigned posting list
        for &centroid_id in &assignments {
            self.posting_lists
                .entry(centroid_id)
                .or_default()
                .push(StoredVector {
                    id: vector_id,
                    data: vector.clone(),
                });
        }

        // Store assignments
        self.vector_assignments.insert(vector_id, assignments);
    }

    /// Recompute assignments for a vector and update posting lists.
    /// Used during reassignment after split/merge.
    fn recompute_vector_assignments(&mut self, vector_id: u64) {
        // Get vector data (don't remove yet in case recompute fails)
        let vector = match self.vector_data.get(&vector_id) {
            Some(v) => v.clone(),
            None => return,
        };

        // Compute new assignments
        let new_assignments = self.compute_multi_assignments(&vector);

        // Get old assignments
        let old_assignments = self.vector_assignments.get(&vector_id).cloned().unwrap_or_default();

        // Remove from posting lists that are no longer assigned
        for centroid_id in old_assignments.difference(&new_assignments) {
            if let Some(posting_list) = self.posting_lists.get_mut(centroid_id) {
                if let Some(pos) = posting_list.iter().position(|v| v.id == vector_id) {
                    posting_list.swap_remove(pos);
                }
            }
        }

        // Add to posting lists that are newly assigned
        for &centroid_id in new_assignments.difference(&old_assignments) {
            self.posting_lists
                .entry(centroid_id)
                .or_default()
                .push(StoredVector {
                    id: vector_id,
                    data: vector.clone(),
                });
        }

        // Update assignments
        self.vector_assignments.insert(vector_id, new_assignments);
    }

    /// Split a centroid into two using k-means
    fn split(&mut self, centroid_id: u64) {
        let vectors = match self.posting_lists.remove(&centroid_id) {
            Some(v) => v,
            None => return,
        };

        if vectors.len() < 2 {
            // Can't split with fewer than 2 vectors
            self.posting_lists.insert(centroid_id, vectors);
            return;
        }

        // Save old centroid before removing (needed for SPFresh reassignment conditions)
        let old_centroid = match self.centroid_vectors.get(&centroid_id) {
            Some(v) => v.clone(),
            None => return,
        };

        // Collect unique vector IDs from this posting (for multi-cluster, track which vectors need recomputation)
        let vector_ids: Vec<u64> = vectors.iter().map(|v| v.id).collect();

        // Remove the old centroid from each vector's assignment set
        for &vid in &vector_ids {
            if let Some(assignments) = self.vector_assignments.get_mut(&vid) {
                assignments.remove(&centroid_id);
            }
        }

        // Run k-means to find 2 new centroids
        let (centroid1, centroid2) = self.kmeans_2(&vectors);
        self.num_splits += 1;

        // Create new centroid IDs
        let new_id1 = self.next_centroid_id;
        let new_id2 = self.next_centroid_id + 1;
        self.next_centroid_id += 2;

        // Remove old centroid from index (soft-delete, ignore errors)
        let _ = self.centroid_index.remove(centroid_id);
        self.centroid_vectors.remove(&centroid_id);

        // Add new centroids to index
        self.add_to_centroids_index(new_id1, &centroid1);
        self.add_to_centroids_index(new_id2, &centroid2);

        // Initialize empty posting lists for new centroids
        self.posting_lists.insert(new_id1, Vec::new());
        self.posting_lists.insert(new_id2, Vec::new());

        // Track vectors that need full reassignment check (SPFresh Condition 1)
        let mut condition1_candidates: Vec<u64> = Vec::new();

        // Assign vectors from the old posting to the closer of the two new centroids
        // This is the same for both single and multi-cluster modes per SPFresh paper
        for v in vectors {
            let dist_to_old = Self::l2_distance(&v.data, &old_centroid);
            let dist1 = Self::l2_distance(&v.data, &centroid1);
            let dist2 = Self::l2_distance(&v.data, &centroid2);

            // Assign to closer of the two new centroids
            let assigned_id = if dist1 <= dist2 { new_id1 } else { new_id2 };
            self.posting_lists.get_mut(&assigned_id).unwrap().push(v.clone());

            // Update vector_assignments: add the new centroid
            // (old centroid was already removed above)
            let assignments = self.vector_assignments.entry(v.id).or_default();
            assignments.insert(assigned_id);

            // Condition 1: if old centroid was closer than BOTH new ones,
            // vector might belong to a neighboring centroid
            if dist_to_old <= dist1 && dist_to_old <= dist2 {
                condition1_candidates.push(v.id);
            }
        }

        // Collect Condition 2 candidates: vectors from neighboring centroids that might need reassignment
        let condition2_candidates = self.collect_condition2_candidates(
            new_id1, new_id2, &old_centroid, &centroid1, &centroid2
        );

        // Merge both conditions and deduplicate
        let mut all_candidates: Vec<u64> = condition1_candidates;
        all_candidates.extend(condition2_candidates);
        all_candidates.sort();
        all_candidates.dedup();

        // Recompute assignments for all candidate vectors (handles both single and multi-cluster)
        for vid in all_candidates {
            let old_assignments = self.vector_assignments.get(&vid).cloned().unwrap_or_default();
            self.recompute_vector_assignments(vid);
            let new_assignments = self.vector_assignments.get(&vid).cloned().unwrap_or_default();

            if old_assignments != new_assignments {
                self.num_reassigned += 1;
            }
        }

        // Check if new centroids need merging (unlikely after split, but for consistency)
        self.check_merge(new_id1);
        self.check_merge(new_id2);
    }

    /// Run k-means with k=2 on the given vectors
    fn kmeans_2(&self, vectors: &[StoredVector]) -> (Vec<f32>, Vec<f32>) {
        let n = vectors.len();
        let dims = self.config.dimensions;

        // Initialize centroids using k-means++ style: pick first randomly, second far from first
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        let idx1 = rng.gen_range(0..n);
        let mut centroid1 = vectors[idx1].data.clone();

        // Pick second centroid weighted by distance from first
        let distances: Vec<f32> = vectors.iter().map(|v| Self::l2_distance(&v.data, &centroid1)).collect();
        let total: f32 = distances.iter().sum();
        let threshold = rng.gen::<f32>() * total;
        let mut cumsum = 0.0;
        let mut idx2 = 0;
        for (i, &d) in distances.iter().enumerate() {
            cumsum += d;
            if cumsum >= threshold {
                idx2 = i;
                break;
            }
        }
        let mut centroid2 = vectors[idx2].data.clone();

        // Run k-means iterations
        for _ in 0..self.config.kmeans_iters {
            // Assign vectors to nearest centroid
            let mut sum1 = vec![0.0f32; dims];
            let mut sum2 = vec![0.0f32; dims];
            let mut count1 = 0usize;
            let mut count2 = 0usize;

            for v in vectors {
                let dist1 = Self::l2_distance(&v.data, &centroid1);
                let dist2 = Self::l2_distance(&v.data, &centroid2);
                if dist1 <= dist2 {
                    for (s, &x) in sum1.iter_mut().zip(v.data.iter()) {
                        *s += x;
                    }
                    count1 += 1;
                } else {
                    for (s, &x) in sum2.iter_mut().zip(v.data.iter()) {
                        *s += x;
                    }
                    count2 += 1;
                }
            }

            // Update centroids
            if count1 > 0 {
                for (c, s) in centroid1.iter_mut().zip(sum1.iter()) {
                    *c = s / count1 as f32;
                }
            }
            if count2 > 0 {
                for (c, s) in centroid2.iter_mut().zip(sum2.iter()) {
                    *c = s / count2 as f32;
                }
            }
        }

        (centroid1, centroid2)
    }

    /// SPFresh Condition 2: Collect vectors from neighboring centroids that might need reassignment.
    ///
    /// For each vector v in a neighboring posting (centroid B), check if:
    ///   min(D(v, A1), D(v, A2)) <= D(v, Ao)
    ///
    /// This means one of the new centroids is closer to v than the old centroid was.
    /// Returns the vector IDs that should be checked for reassignment.
    ///
    /// Per the SPFresh paper: "LIRE only examines nearby postings for reassignment check
    /// by selecting several Ao's nearest postings" - we find neighbors of the OLD centroid.
    fn collect_condition2_candidates(
        &self,
        new_id1: u64,
        new_id2: u64,
        old_centroid: &[f32],
        centroid1: &[f32],
        centroid2: &[f32],
    ) -> Vec<u64> {
        // Find neighbors of the OLD centroid position (Ao)
        // Note: Ao has been removed, but we search using its vector position.
        // This will return the nearest centroids to where Ao was, including A1 and A2.
        let neighbors = self
            .centroid_index
            .search(old_centroid, self.config.reassign_neighbors + 2)  // +2 to account for A1, A2
            .expect("Search failed");

        // Filter out the new centroids (A1 and A2) - we only want the neighboring postings
        let neighbor_ids: Vec<u64> = neighbors.keys.iter()
            .copied()
            .filter(|&id| id != new_id1 && id != new_id2)
            .collect();

        // Collect vector IDs that need reassignment check
        let mut candidates: Vec<u64> = Vec::new();

        // Check each neighbor's posting list
        for neighbor_id in neighbor_ids {
            if let Some(posting_list) = self.posting_lists.get(&neighbor_id) {
                for v in posting_list {
                    let dist_to_old = Self::l2_distance(&v.data, old_centroid);
                    let dist_to_new1 = Self::l2_distance(&v.data, centroid1);
                    let dist_to_new2 = Self::l2_distance(&v.data, centroid2);
                    let min_dist_to_new = dist_to_new1.min(dist_to_new2);

                    // Condition 2: if either new centroid is closer than the old centroid,
                    // this vector might need reassignment
                    if min_dist_to_new <= dist_to_old {
                        candidates.push(v.id);
                    }
                }
            }
        }

        candidates
    }

    /// Reassign a batch of vectors to their true nearest centroids
    /// Returns vectors grouped by their new centroid assignment
    fn find_assignments(&self, vectors: Vec<StoredVector>) -> HashMap<u64, Vec<StoredVector>> {
        let mut assignments: HashMap<u64, Vec<StoredVector>> = HashMap::new();

        for v in vectors {
            let nearest = self.centroid_index.search(&v.data, 1).expect("Search failed");
            let nearest_centroid = nearest.keys[0];
            assignments.entry(nearest_centroid).or_default().push(v);
        }

        assignments
    }

    /// Check if a centroid needs to be merged
    fn check_merge(&mut self, centroid_id: u64) {
        let size = self.posting_lists.get(&centroid_id).map(|l| l.len()).unwrap_or(0);

        // Don't merge if above threshold or if it's the only centroid
        if size >= self.config.min_posting_size || self.centroid_vectors.len() <= 1 {
            return;
        }

        self.merge(centroid_id);
    }

    /// Merge a centroid by reassigning its vectors to their true nearest centroids
    fn merge(&mut self, centroid_id: u64) {
        if !self.centroid_vectors.contains_key(&centroid_id) {
            return;
        }

        // Remove this centroid from the index first
        let vectors = self.posting_lists.remove(&centroid_id).unwrap_or_default();
        let _ = self.centroid_index.remove(centroid_id);
        self.centroid_vectors.remove(&centroid_id);

        if vectors.is_empty() {
            return;
        }

        self.num_merges += 1;

        // Collect unique vector IDs and remove the merged centroid from their assignments
        let vector_ids: Vec<u64> = vectors.iter().map(|v| v.id).collect();
        for &vid in &vector_ids {
            if let Some(assignments) = self.vector_assignments.get_mut(&vid) {
                assignments.remove(&centroid_id);
            }
        }

        // Recompute assignments for each vector
        let mut num_reassigned = 0;
        for vid in vector_ids {
            let old_assignments = self.vector_assignments.get(&vid).cloned().unwrap_or_default();
            self.recompute_vector_assignments(vid);
            let new_assignments = self.vector_assignments.get(&vid).cloned().unwrap_or_default();

            if old_assignments != new_assignments {
                num_reassigned += 1;
            }
        }

        self.num_reassigned += num_reassigned;

        // Note: We don't check for splits here to avoid cycles.
        // Splits will happen naturally on the next insert if needed.
    }

    /// Compute centroid (mean) of vectors
    fn compute_centroid(&self, vectors: &[StoredVector]) -> Vec<f32> {
        let dims = self.config.dimensions;
        let mut sum = vec![0.0f32; dims];

        for v in vectors {
            for (s, &x) in sum.iter_mut().zip(v.data.iter()) {
                *s += x;
            }
        }

        let count = vectors.len() as f32;
        for s in sum.iter_mut() {
            *s /= count;
        }

        sum
    }

    /// Search for k nearest neighbors
    /// Uses query-aware dynamic pruning from SPANN: only search posting lists where
    /// dist(q, centroid) <= (1 + epsilon) * dist(q, closest_centroid)
    pub fn search(&self, query: &[f32], k: usize) -> SearchResult {
        assert_eq!(query.len(), self.config.dimensions, "Query dimension mismatch");

        // Find top centroids to probe
        let num_probes = self.config.num_probes.min(self.centroid_vectors.len());
        let centroid_results = self.centroid_index.search(query, num_probes).expect("Search failed");

        // Collect all candidate vectors from probed posting lists
        let mut candidates: Vec<(u64, f32)> = Vec::new();
        let mut probes_searched = 0;

        // Apply query-aware dynamic pruning (SPANN optimization)
        // Only search posting lists where: dist <= (1 + epsilon) * min_dist
        let pruning_enabled = self.config.pruning_threshold > 0.0;
        let min_dist = if pruning_enabled && !centroid_results.distances.is_empty() {
            centroid_results.distances[0]
        } else {
            0.0
        };
        let dist_threshold = if pruning_enabled {
            (1.0 + self.config.pruning_threshold) * min_dist
        } else {
            f32::MAX
        };

        for (i, &centroid_id) in centroid_results.keys.iter().enumerate() {
            // Dynamic pruning: skip posting lists that are too far from the query
            if pruning_enabled && centroid_results.distances[i] > dist_threshold {
                continue;
            }

            if let Some(posting_list) = self.posting_lists.get(&centroid_id) {
                probes_searched += 1;
                for v in posting_list {
                    let dist = Self::l2_distance(query, &v.data);
                    candidates.push((v.id, dist));
                }
            }
        }

        // Deduplicate candidates (same vector may appear in multiple posting lists with multi-cluster)
        // Keep the entry with the smallest distance for each vector ID
        let mut seen: HashMap<u64, f32> = HashMap::new();
        for (id, dist) in candidates {
            seen.entry(id)
                .and_modify(|d| *d = d.min(dist))
                .or_insert(dist);
        }
        let mut deduped: Vec<(u64, f32)> = seen.into_iter().collect();

        // Sort by distance and return top k
        deduped.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        deduped.truncate(k);
        SearchResult {
            neighbors: deduped,
            probes_searched,
        }
    }

    /// Brute force search (for computing recall)
    pub fn brute_force_search(&self, query: &[f32], k: usize) -> Vec<(u64, f32)> {
        // Use vector_data for canonical deduplication if available, otherwise fall back to posting lists
        let all_vectors: Vec<(u64, f32)> = if !self.vector_data.is_empty() {
            self.vector_data.iter()
                .map(|(&id, data)| (id, Self::l2_distance(query, data)))
                .collect()
        } else {
            // Deduplicate from posting lists
            let mut seen: HashMap<u64, f32> = HashMap::new();
            for posting_list in self.posting_lists.values() {
                for v in posting_list {
                    let dist = Self::l2_distance(query, &v.data);
                    seen.entry(v.id)
                        .and_modify(|d| *d = d.min(dist))
                        .or_insert(dist);
                }
            }
            seen.into_iter().collect()
        };

        let mut sorted = all_vectors;
        sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        sorted.truncate(k);
        sorted
    }

    /// Compute L2 squared distance
    fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y) * (x - y))
            .sum()
    }

    /// Get the number of dimensions
    pub fn dimensions(&self) -> usize {
        self.config.dimensions
    }

    /// Get the configuration
    pub fn config(&self) -> &SPFreshConfig {
        &self.config
    }

    /// Set the number of probes (for tuning recall after loading)
    pub fn set_num_probes(&mut self, num_probes: usize) {
        self.config.num_probes = num_probes;
    }

    /// Set the HNSW expansion factor for search (for tuning recall after loading)
    pub fn set_expansion_search(&mut self, expansion_search: usize) {
        self.config.expansion_search = expansion_search;
        self.centroid_index.change_expansion_search(expansion_search);
    }

    /// Set the pruning threshold (for tuning recall/performance after loading)
    pub fn set_pruning_threshold(&mut self, pruning_threshold: f32) {
        self.config.pruning_threshold = pruning_threshold;
    }

    /// Get index statistics
    pub fn stats(&self) -> IndexStats {
        let posting_sizes: Vec<usize> = self.posting_lists.values().map(|l| l.len()).collect();
        let min_size = posting_sizes.iter().min().copied().unwrap_or(0);
        let max_size = posting_sizes.iter().max().copied().unwrap_or(0);
        let total_posting_entries: usize = posting_sizes.iter().sum();
        let avg_size = if posting_sizes.is_empty() {
            0.0
        } else {
            total_posting_entries as f64 / posting_sizes.len() as f64
        };

        // Replication factor: avg number of posting lists per vector
        let replication_factor = if self.total_vectors > 0 {
            total_posting_entries as f64 / self.total_vectors as f64
        } else {
            1.0
        };

        IndexStats {
            num_centroids: self.centroid_vectors.len(),
            total_vectors: self.total_vectors,
            total_posting_entries,
            min_posting_size: min_size,
            max_posting_size: max_size,
            avg_posting_size: avg_size,
            replication_factor,
            num_splits: self.num_splits,
            num_merges: self.num_merges,
            num_reassigned: self.num_reassigned,
        }
    }

    /// Analyze compression ratio of random posting lists using zstd
    /// Returns (num_postings_sampled, total_original_bytes, total_compressed_bytes, compression_ratio)
    pub fn analyze_compression(&self, num_samples: usize, compression_level: i32) -> CompressionStats {
        use rand::seq::SliceRandom;

        let centroid_ids: Vec<u64> = self.posting_lists.keys().copied().collect();
        let num_samples = num_samples.min(centroid_ids.len());

        if num_samples == 0 {
            return CompressionStats {
                num_postings: 0,
                total_vectors: 0,
                original_bytes: 0,
                compressed_bytes: 0,
                compression_ratio: 1.0,
            };
        }

        // Sample random posting lists
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        let mut sampled_ids = centroid_ids.clone();
        sampled_ids.shuffle(&mut rng);
        sampled_ids.truncate(num_samples);

        let mut total_original: usize = 0;
        let mut total_compressed: usize = 0;
        let mut total_vectors: usize = 0;

        for centroid_id in &sampled_ids {
            if let Some(posting_list) = self.posting_lists.get(centroid_id) {
                // Serialize the posting list to bytes (vector IDs + f32 data)
                let mut raw_bytes: Vec<u8> = Vec::new();

                for sv in posting_list {
                    // Write vector ID (8 bytes)
                    raw_bytes.extend_from_slice(&sv.id.to_le_bytes());
                    // Write vector data (dimensions * 4 bytes)
                    for &val in &sv.data {
                        raw_bytes.extend_from_slice(&val.to_le_bytes());
                    }
                }

                let original_size = raw_bytes.len();
                total_original += original_size;
                total_vectors += posting_list.len();

                // Compress with zstd
                let compressed = zstd::encode_all(raw_bytes.as_slice(), compression_level)
                    .expect("zstd compression failed");
                total_compressed += compressed.len();
            }
        }

        let compression_ratio = if total_compressed > 0 {
            total_original as f64 / total_compressed as f64
        } else {
            1.0
        };

        CompressionStats {
            num_postings: num_samples,
            total_vectors,
            original_bytes: total_original,
            compressed_bytes: total_compressed,
            compression_ratio,
        }
    }

    /// Save the index to a directory
    pub fn save<P: AsRef<Path>>(&self, dir: P) -> std::io::Result<()> {
        let dir = dir.as_ref();
        fs::create_dir_all(dir)?;

        // Save centroid index using usearch's native format
        let centroid_path = dir.join("centroids.usearch");
        self.centroid_index
            .save(centroid_path.to_str().unwrap())
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.what()))?;

        // Save centroid vectors as binary (for reconstruction)
        let centroid_vectors_path = dir.join("centroid_vectors.bin");
        self.save_centroid_vectors(&centroid_vectors_path)?;

        // Save posting lists (all vectors with their assignments)
        let vectors_path = dir.join("vectors.bin");
        self.save_vectors(&vectors_path)?;

        // Save metadata as JSON
        let metadata_path = dir.join("metadata.json");
        self.save_metadata(&metadata_path)?;

        Ok(())
    }

    /// Save centroid vectors as binary
    fn save_centroid_vectors<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        // Write header: num_centroids, dimensions
        let num_centroids = self.centroid_vectors.len() as u64;
        let dimensions = self.config.dimensions as u64;
        writer.write_all(&num_centroids.to_le_bytes())?;
        writer.write_all(&dimensions.to_le_bytes())?;

        // Write each centroid: id (u64), vector (f32 * dimensions)
        for (&id, vector) in &self.centroid_vectors {
            writer.write_all(&id.to_le_bytes())?;
            for &val in vector {
                writer.write_all(&val.to_le_bytes())?;
            }
        }

        writer.flush()?;
        Ok(())
    }

    /// Save all vectors with their centroid assignments as binary
    /// With multi-cluster assignment, the same vector may appear in multiple posting lists
    fn save_vectors<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        // Count total posting entries (includes duplicates for multi-cluster)
        let total_posting_entries: u64 = self.posting_lists.values()
            .map(|l| l.len() as u64)
            .sum();
        let dimensions = self.config.dimensions as u64;

        // Write header: total_posting_entries, dimensions, total_unique_vectors
        writer.write_all(&total_posting_entries.to_le_bytes())?;
        writer.write_all(&dimensions.to_le_bytes())?;
        writer.write_all(&(self.total_vectors as u64).to_le_bytes())?;

        // Write each posting entry: centroid_id (u64), vector_id (u64), vector (f32 * dimensions)
        for (&centroid_id, posting_list) in &self.posting_lists {
            for stored in posting_list {
                writer.write_all(&centroid_id.to_le_bytes())?;
                writer.write_all(&stored.id.to_le_bytes())?;
                for &val in &stored.data {
                    writer.write_all(&val.to_le_bytes())?;
                }
            }
        }

        writer.flush()?;
        Ok(())
    }

    /// Save metadata as JSON
    fn save_metadata<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        let stats = self.stats();
        let metadata = format!(
            r#"{{
  "dimensions": {},
  "max_posting_size": {},
  "min_posting_size": {},
  "num_probes": {},
  "reassign_neighbors": {},
  "kmeans_iters": {},
  "connectivity": {},
  "expansion_add": {},
  "expansion_search": {},
  "pruning_threshold": {},
  "assignment_threshold": {},
  "max_assignment_count": {},
  "num_centroids": {},
  "total_vectors": {},
  "total_posting_entries": {},
  "replication_factor": {:.2},
  "num_splits": {},
  "num_merges": {},
  "num_reassigned": {},
  "min_posting_size_actual": {},
  "max_posting_size_actual": {},
  "avg_posting_size": {:.2}
}}
"#,
            self.config.dimensions,
            self.config.max_posting_size,
            self.config.min_posting_size,
            self.config.num_probes,
            self.config.reassign_neighbors,
            self.config.kmeans_iters,
            self.config.connectivity,
            self.config.expansion_add,
            self.config.expansion_search,
            self.config.pruning_threshold,
            self.config.assignment_threshold,
            self.config.max_assignment_count,
            stats.num_centroids,
            stats.total_vectors,
            stats.total_posting_entries,
            stats.replication_factor,
            stats.num_splits,
            stats.num_merges,
            stats.num_reassigned,
            stats.min_posting_size,
            stats.max_posting_size,
            stats.avg_posting_size
        );

        writer.write_all(metadata.as_bytes())?;
        writer.flush()?;
        Ok(())
    }

    /// Load an index from a directory
    pub fn load<P: AsRef<Path>>(dir: P) -> std::io::Result<Self> {
        let dir = dir.as_ref();

        // Load metadata first to get config
        let metadata_path = dir.join("metadata.json");
        let metadata_content = fs::read_to_string(&metadata_path)?;
        let config = Self::parse_metadata(&metadata_content)?;

        // Create index options
        let index_options = IndexOptions {
            dimensions: config.dimensions,
            metric: MetricKind::L2sq,
            quantization: ScalarKind::F32,
            connectivity: config.connectivity,
            expansion_add: config.expansion_add,
            expansion_search: config.expansion_search,
            multi: false,
        };

        // Load centroid index using usearch's native format
        let centroid_path = dir.join("centroids.usearch");
        let centroid_index = new_index(&index_options)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.what()))?;
        centroid_index
            .load(centroid_path.to_str().unwrap())
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.what()))?;

        // Load centroid vectors
        let centroid_vectors_path = dir.join("centroid_vectors.bin");
        let centroid_vectors = Self::load_centroid_vectors(&centroid_vectors_path)?;

        // Load posting lists (all vectors with their assignments)
        let vectors_path = dir.join("vectors.bin");
        let (posting_lists, total_vectors, next_centroid_id) =
            Self::load_vectors(&vectors_path, config.dimensions)?;

        // Reconstruct vector_assignments and vector_data from posting lists
        let mut vector_assignments: HashMap<u64, HashSet<u64>> = HashMap::new();
        let mut vector_data: HashMap<u64, Vec<f32>> = HashMap::new();
        for (&centroid_id, posting_list) in &posting_lists {
            for v in posting_list {
                vector_assignments.entry(v.id).or_default().insert(centroid_id);
                // Store vector data (may overwrite with same data if multi-cluster)
                vector_data.entry(v.id).or_insert_with(|| v.data.clone());
            }
        }

        Ok(Self {
            config,
            centroid_index,
            posting_lists,
            centroid_vectors,
            vector_assignments,
            vector_data,
            next_centroid_id,
            total_vectors,
            num_splits: 0,
            num_merges: 0,
            num_reassigned: 0,
        })
    }

    /// Parse metadata JSON to extract config
    fn parse_metadata(content: &str) -> std::io::Result<SPFreshConfig> {
        // Simple JSON parsing without serde
        let get_value = |key: &str| -> std::io::Result<usize> {
            let pattern = format!("\"{}\": ", key);
            let start = content.find(&pattern).ok_or_else(|| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, format!("Missing key: {}", key))
            })?;
            let rest = &content[start + pattern.len()..];
            let end = rest.find(|c: char| !c.is_ascii_digit()).unwrap_or(rest.len());
            rest[..end].parse().map_err(|_| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, format!("Invalid value for {}", key))
            })
        };

        // Parse float values (handles digits, decimal point, and negative sign)
        let get_float_value = |key: &str| -> std::io::Result<f32> {
            let pattern = format!("\"{}\": ", key);
            let start = content.find(&pattern).ok_or_else(|| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, format!("Missing key: {}", key))
            })?;
            let rest = &content[start + pattern.len()..];
            let end = rest.find(|c: char| !c.is_ascii_digit() && c != '.' && c != '-').unwrap_or(rest.len());
            rest[..end].parse().map_err(|_| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, format!("Invalid value for {}", key))
            })
        };

        // Optional values with defaults (for backward compatibility)
        let get_value_or_default = |key: &str, default: usize| -> usize {
            get_value(key).unwrap_or(default)
        };

        let get_float_or_default = |key: &str, default: f32| -> f32 {
            get_float_value(key).unwrap_or(default)
        };

        Ok(SPFreshConfig {
            dimensions: get_value("dimensions")?,
            max_posting_size: get_value("max_posting_size")?,
            min_posting_size: get_value("min_posting_size")?,
            num_probes: get_value("num_probes")?,
            reassign_neighbors: get_value("reassign_neighbors")?,
            kmeans_iters: get_value("kmeans_iters")?,
            connectivity: get_value_or_default("connectivity", 16),
            expansion_add: get_value_or_default("expansion_add", 128),
            expansion_search: get_value_or_default("expansion_search", 64),
            pruning_threshold: get_float_or_default("pruning_threshold", 0.0),
            assignment_threshold: get_float_or_default("assignment_threshold", 0.0),
            max_assignment_count: get_value_or_default("max_assignment_count", 0),
        })
    }

    /// Load centroid vectors from binary file
    fn load_centroid_vectors<P: AsRef<Path>>(path: P) -> std::io::Result<HashMap<u64, Vec<f32>>> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        // Read header
        let mut buf8 = [0u8; 8];
        reader.read_exact(&mut buf8)?;
        let num_centroids = u64::from_le_bytes(buf8) as usize;
        reader.read_exact(&mut buf8)?;
        let dimensions = u64::from_le_bytes(buf8) as usize;

        let mut centroid_vectors = HashMap::with_capacity(num_centroids);
        let mut buf4 = [0u8; 4];

        for _ in 0..num_centroids {
            // Read centroid ID
            reader.read_exact(&mut buf8)?;
            let id = u64::from_le_bytes(buf8);

            // Read vector
            let mut vector = Vec::with_capacity(dimensions);
            for _ in 0..dimensions {
                reader.read_exact(&mut buf4)?;
                vector.push(f32::from_le_bytes(buf4));
            }
            centroid_vectors.insert(id, vector);
        }

        Ok(centroid_vectors)
    }

    /// Load vectors from binary file, returning posting lists and metadata
    fn load_vectors<P: AsRef<Path>>(
        path: P,
        dimensions: usize,
    ) -> std::io::Result<(HashMap<u64, Vec<StoredVector>>, usize, u64)> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        // Read header: total_posting_entries, dimensions, total_unique_vectors
        let mut buf8 = [0u8; 8];
        reader.read_exact(&mut buf8)?;
        let total_posting_entries = u64::from_le_bytes(buf8) as usize;
        reader.read_exact(&mut buf8)?;
        let file_dimensions = u64::from_le_bytes(buf8) as usize;
        reader.read_exact(&mut buf8)?;
        let total_vectors = u64::from_le_bytes(buf8) as usize;

        if file_dimensions != dimensions {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Dimension mismatch: expected {}, got {}", dimensions, file_dimensions),
            ));
        }

        let mut posting_lists: HashMap<u64, Vec<StoredVector>> = HashMap::new();
        let mut max_centroid_id: u64 = 0;
        let mut buf4 = [0u8; 4];

        // Read all posting entries (may include same vector in multiple lists)
        for _ in 0..total_posting_entries {
            // Read centroid ID
            reader.read_exact(&mut buf8)?;
            let centroid_id = u64::from_le_bytes(buf8);
            max_centroid_id = max_centroid_id.max(centroid_id);

            // Read vector ID
            reader.read_exact(&mut buf8)?;
            let vector_id = u64::from_le_bytes(buf8);

            // Read vector data
            let mut data = Vec::with_capacity(dimensions);
            for _ in 0..dimensions {
                reader.read_exact(&mut buf4)?;
                data.push(f32::from_le_bytes(buf4));
            }

            posting_lists
                .entry(centroid_id)
                .or_default()
                .push(StoredVector { id: vector_id, data });
        }

        // Next centroid ID should be max + 1
        let next_centroid_id = max_centroid_id + 1;

        Ok((posting_lists, total_vectors, next_centroid_id))
    }
}

/// Index statistics
#[derive(Debug, Clone)]
pub struct IndexStats {
    pub num_centroids: usize,
    pub total_vectors: usize,
    pub total_posting_entries: usize,
    pub min_posting_size: usize,
    pub max_posting_size: usize,
    pub avg_posting_size: f64,
    pub replication_factor: f64,
    pub num_splits: usize,
    pub num_merges: usize,
    pub num_reassigned: usize,
}

/// Compression statistics for posting lists
pub struct CompressionStats {
    pub num_postings: usize,
    pub total_vectors: usize,
    pub original_bytes: usize,
    pub compressed_bytes: usize,
    pub compression_ratio: f64,
}

/// Search result with statistics
pub struct SearchResult {
    /// The k nearest neighbors (id, distance)
    pub neighbors: Vec<(u64, f32)>,
    /// Number of posting lists actually searched
    pub probes_searched: usize,
}

/// CLI arguments
#[derive(Parser, Debug)]
#[command(name = "spfresh")]
#[command(about = "SPFresh-based vector index for tuning recall")]
struct Args {
    /// Number of vectors to insert
    #[arg(short = 'n', long, default_value_t = 100000)]
    num_vectors: usize,

    /// Vector dimensions
    #[arg(short = 'd', long, default_value_t = 128)]
    dimensions: usize,

    /// Maximum posting list size before split (N)
    #[arg(long, default_value_t = 1000)]
    max_posting_size: usize,

    /// Minimum posting list size before merge (M)
    #[arg(long, default_value_t = 100)]
    min_posting_size: usize,

    /// Number of centroids to probe during search
    #[arg(long, default_value_t = 10)]
    num_probes: usize,

    /// Number of neighbors to check for reassignment
    #[arg(long, default_value_t = 5)]
    reassign_neighbors: usize,

    /// Number of k-means iterations during split
    #[arg(long, default_value_t = 10)]
    kmeans_iters: usize,

    /// HNSW connectivity (M parameter) - edges per node in centroid index
    #[arg(long, default_value_t = 16)]
    connectivity: usize,

    /// HNSW expansion factor during index construction
    #[arg(long, default_value_t = 128)]
    expansion_add: usize,

    /// HNSW expansion factor during search
    #[arg(long, default_value_t = 64)]
    expansion_search: usize,

    /// Query-aware dynamic pruning threshold (epsilon from SPANN paper)
    /// Only search posting lists where: dist(q, centroid) <= (1 + epsilon) * dist(q, closest)
    /// Set to 0.0 to disable (default). Try 0.5-2.0 for recall@10, 0.1-0.6 for recall@1
    #[arg(long, default_value_t = 0.0)]
    pruning_threshold: f32,

    /// Multi-cluster assignment threshold (epsilon1 from SPANN paper)
    /// Assign vector to multiple clusters where: dist(v, c) <= (1 + epsilon) * dist(v, closest)
    /// Set to 0.0 to disable (default, single assignment). Try 0.1-0.5 for better recall
    #[arg(long, default_value_t = 0.0)]
    assignment_threshold: f32,

    /// Maximum number of centroids a vector can be assigned to (caps replication)
    /// Set to 0 for unlimited (default). Typical values: 4-16
    #[arg(long, default_value_t = 0)]
    max_assignment_count: usize,

    /// Number of queries for recall evaluation
    #[arg(long, default_value_t = 100)]
    num_queries: usize,

    /// K for k-NN search
    #[arg(short = 'k', long, default_value_t = 10)]
    k: usize,

    /// Random seed
    #[arg(short = 's', long, default_value_t = 42)]
    seed: u64,

    /// Output directory to save index and vectors (if specified)
    #[arg(short = 'o', long)]
    output_dir: Option<String>,

    /// Input directory to load a previously saved index (skips building)
    #[arg(short = 'i', long)]
    input_dir: Option<String>,

    /// Dataset type: random, sift1m, sift100m, or deep1m
    #[arg(long, default_value = "random")]
    dataset: DatasetType,

    /// Directory containing dataset files (required for sift1m/sift100m/deep1m)
    #[arg(long)]
    dataset_dir: Option<String>,

    /// Show download instructions for a dataset
    #[arg(long)]
    download_help: Option<DatasetType>,
}

fn generate_random_vectors(num_vectors: usize, dimensions: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    (0..num_vectors)
        .map(|_| (0..dimensions).map(|_| rng.gen::<f32>()).collect())
        .collect()
}

fn compute_recall(approximate: &[(u64, f32)], exact: &[(u64, f32)]) -> f64 {
    let exact_ids: HashSet<u64> = exact.iter().map(|(id, _)| *id).collect();
    let hits = approximate.iter().filter(|(id, _)| exact_ids.contains(id)).count();
    hits as f64 / exact.len() as f64
}

/// Compute recall using ground truth indices (for benchmark datasets)
fn compute_recall_gt(approximate: &[(u64, f32)], ground_truth: &[u32], k: usize) -> f64 {
    let gt_set: HashSet<u64> = ground_truth.iter().take(k).map(|&id| id as u64).collect();
    let hits = approximate.iter().filter(|(id, _)| gt_set.contains(id)).count();
    hits as f64 / k as f64
}

fn format_bytes(bytes: usize) -> String {
    if bytes >= 1024 * 1024 * 1024 {
        format!("{:.2} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    } else if bytes >= 1024 * 1024 {
        format!("{:.2} MB", bytes as f64 / (1024.0 * 1024.0))
    } else if bytes >= 1024 {
        format!("{:.2} KB", bytes as f64 / 1024.0)
    } else {
        format!("{} B", bytes)
    }
}

fn main() {
    let args = Args::parse();

    // Handle download help
    if let Some(dataset_type) = args.download_help {
        datasets::print_download_instructions(dataset_type);
        return;
    }

    // Load dataset if using sift1m or deep1m
    let dataset: Option<Dataset> = if args.dataset != DatasetType::Random {
        let dataset_dir = args.dataset_dir.as_ref().unwrap_or_else(|| {
            eprintln!("Error: --dataset-dir is required when using {} dataset", args.dataset);
            eprintln!("Use --download-help {} to see download instructions", args.dataset);
            std::process::exit(1);
        });

        match datasets::load_dataset(args.dataset, Path::new(dataset_dir), Some(args.num_vectors)) {
            Ok(ds) => Some(ds),
            Err(e) => {
                eprintln!("Error loading dataset: {}", e);
                eprintln!("Use --download-help {} to see download instructions", args.dataset);
                std::process::exit(1);
            }
        }
    } else {
        None
    };

    // Get dimensions from dataset or args
    let dimensions = dataset.as_ref().map(|ds| ds.dimensions).unwrap_or(args.dimensions);

    let config = SPFreshConfig {
        dimensions,
        max_posting_size: args.max_posting_size,
        min_posting_size: args.min_posting_size,
        num_probes: args.num_probes,
        reassign_neighbors: args.reassign_neighbors,
        kmeans_iters: args.kmeans_iters,
        connectivity: args.connectivity,
        expansion_add: args.expansion_add,
        expansion_search: args.expansion_search,
        pruning_threshold: args.pruning_threshold,
        assignment_threshold: args.assignment_threshold,
        max_assignment_count: args.max_assignment_count,
    };

    println!("=== SPFresh Index Benchmark ===");
    println!();

    // Either load existing index or build new one
    let index = if let Some(ref input_dir) = args.input_dir {
        // Load from directory
        println!("--- Loading Index ---");
        print!("Loading from {}... ", input_dir);
        std::io::Write::flush(&mut std::io::stdout()).unwrap();
        let start = Instant::now();

        match SPFreshIndex::load(input_dir) {
            Ok(mut idx) => {
                println!("done ({:.2}s)", start.elapsed().as_secs_f64());

                // Print saved configuration
                let saved_config = idx.config();
                println!();
                println!("Saved index configuration:");
                println!("  Dimensions:         {:>10}", saved_config.dimensions);
                println!("  Max posting size:   {:>10}", saved_config.max_posting_size);
                println!("  Min posting size:   {:>10}", saved_config.min_posting_size);
                println!("  Reassign neighbors: {:>10}", saved_config.reassign_neighbors);
                println!("  K-means iters:      {:>10}", saved_config.kmeans_iters);
                println!("  Connectivity:       {:>10}", saved_config.connectivity);
                println!("  Expansion add:      {:>10}", saved_config.expansion_add);
                println!("  Pruning (saved):    {:>10.2}", saved_config.pruning_threshold);
                println!("  Assignment (saved): {:>10.2}", saved_config.assignment_threshold);
                println!("  Max assign (saved): {:>10}", if saved_config.max_assignment_count == 0 { "unlimited".to_string() } else { saved_config.max_assignment_count.to_string() });

                // Override search parameters from command line
                idx.set_num_probes(args.num_probes);
                idx.set_expansion_search(args.expansion_search);
                idx.set_pruning_threshold(args.pruning_threshold);

                let stats = idx.stats();
                println!();
                println!("Index stats:");
                println!("  Num centroids:      {:>10}", stats.num_centroids);
                println!("  Total vectors:      {:>10}", stats.total_vectors);
                println!("  Posting entries:    {:>10}", stats.total_posting_entries);
                println!("  Replication factor: {:>10.2}", stats.replication_factor);
                println!("  Min posting size:   {:>10}", stats.min_posting_size);
                println!("  Max posting size:   {:>10}", stats.max_posting_size);
                println!("  Avg posting size:   {:>10.1}", stats.avg_posting_size);
                println!();
                println!("Search parameters (from args):");
                println!("  Num probes:         {:>10}", args.num_probes);
                println!("  Expansion search:   {:>10}", args.expansion_search);
                println!("  Pruning threshold:  {:>10.2}", args.pruning_threshold);
                idx
            }
            Err(e) => {
                println!("FAILED: {}", e);
                std::process::exit(1);
            }
        }
    } else {
        // Build new index
        let num_vectors = dataset.as_ref().map(|ds| ds.num_base()).unwrap_or(args.num_vectors);

        println!("Configuration:");
        println!("  Dataset:            {:>10}", args.dataset);
        println!("  Vectors:            {:>10}", num_vectors);
        println!("  Dimensions:         {:>10}", dimensions);
        println!("  Max posting size:   {:>10}", args.max_posting_size);
        println!("  Min posting size:   {:>10}", args.min_posting_size);
        println!("  Num probes:         {:>10}", args.num_probes);
        println!("  Reassign neighbors: {:>10}", args.reassign_neighbors);
        println!("  K-means iters:      {:>10}", args.kmeans_iters);
        println!("  Connectivity:       {:>10}", args.connectivity);
        println!("  Expansion add:      {:>10}", args.expansion_add);
        println!("  Expansion search:   {:>10}", args.expansion_search);
        println!("  Pruning threshold:  {:>10.2}", args.pruning_threshold);
        println!("  Assignment thresh:  {:>10.2}", args.assignment_threshold);
        println!("  Max assignments:    {:>10}", if args.max_assignment_count == 0 { "unlimited".to_string() } else { args.max_assignment_count.to_string() });
        println!("  Num queries:        {:>10}", args.num_queries);
        println!("  K (for k-NN):       {:>10}", args.k);
        println!("  Seed:               {:>10}", args.seed);
        println!();

        // Get vectors from dataset or generate random
        let vectors: Vec<Vec<f32>> = if let Some(ref ds) = dataset {
            println!("Using {} dataset vectors", ds.name);
            ds.base_vectors.clone()
        } else {
            print!("Generating {} vectors... ", num_vectors);
            std::io::Write::flush(&mut std::io::stdout()).unwrap();
            let start = Instant::now();
            let vecs = generate_random_vectors(num_vectors, dimensions, args.seed);
            println!("done ({:.2}s)", start.elapsed().as_secs_f64());
            vecs
        };

        // Build index
        println!();
        println!("--- Building Index ---");
        let mut idx = SPFreshIndex::new(config);
        let start = Instant::now();

        let report_interval = (num_vectors / 20).max(1000).min(num_vectors);
        let mut last_report = Instant::now();

        for (i, vector) in vectors.iter().enumerate() {
            idx.insert(i as u64, vector);

            // Report every interval or every 10 seconds, whichever comes first
            let should_report = (i + 1) % report_interval == 0
                || last_report.elapsed().as_secs_f64() >= 1.0
                || i + 1 == num_vectors;

            if should_report && i > 0 {
                let elapsed = start.elapsed().as_secs_f64();
                let rate = (i + 1) as f64 / elapsed;
                let stats = idx.stats();

                // Use carriage return to overwrite the line
                println!(
                    "{:>6.1}% | {:>8} vecs | {:>5} centroids | {:>5} splits | {:>5} merges | {:>7} reassigned | {:>8.0} vec/s | {:.1}s",
                    (i + 1) as f64 / num_vectors as f64 * 100.0,
                    i + 1,
                    stats.num_centroids,
                    stats.num_splits,
                    stats.num_merges,
                    stats.num_reassigned,
                    rate,
                    elapsed
                );
                last_report = Instant::now();
            }
        }
        let build_time = start.elapsed();
        println!(); // New line after progress

        println!();
        let stats = idx.stats();
        println!("Build complete in {:.2}s ({:.0} vectors/sec)", build_time.as_secs_f64(), num_vectors as f64 / build_time.as_secs_f64());
        println!("  Num centroids:      {:>10}", stats.num_centroids);
        println!("  Total vectors:      {:>10}", stats.total_vectors);
        println!("  Posting entries:    {:>10}", stats.total_posting_entries);
        println!("  Replication factor: {:>10.2}", stats.replication_factor);
        println!("  Min posting size:   {:>10}", stats.min_posting_size);
        println!("  Max posting size:   {:>10}", stats.max_posting_size);
        println!("  Avg posting size:   {:>10.1}", stats.avg_posting_size);
        println!("  Total splits:       {:>10}", stats.num_splits);
        println!("  Total merges:       {:>10}", stats.num_merges);
        println!("  Total reassigned:   {:>10}", stats.num_reassigned);
        idx
    };

    // Save index if output directory specified
    if let Some(ref output_dir) = args.output_dir {
        println!();
        println!("--- Saving Index ---");
        print!("Saving to {}... ", output_dir);
        std::io::Write::flush(&mut std::io::stdout()).unwrap();
        let start = Instant::now();
        match index.save(output_dir) {
            Ok(()) => {
                println!("done ({:.2}s)", start.elapsed().as_secs_f64());
                // Print file sizes
                let dir = Path::new(output_dir);
                if let Ok(metadata) = fs::metadata(dir.join("centroids.usearch")) {
                    println!("  centroids.usearch:  {:>10}", format_bytes(metadata.len() as usize));
                }
                if let Ok(metadata) = fs::metadata(dir.join("centroid_vectors.bin")) {
                    println!("  centroid_vectors.bin: {:>8}", format_bytes(metadata.len() as usize));
                }
                if let Ok(metadata) = fs::metadata(dir.join("vectors.bin")) {
                    println!("  vectors.bin:        {:>10}", format_bytes(metadata.len() as usize));
                }
                if let Ok(metadata) = fs::metadata(dir.join("metadata.json")) {
                    println!("  metadata.json:      {:>10}", format_bytes(metadata.len() as usize));
                }
            }
            Err(e) => {
                println!("FAILED: {}", e);
            }
        }
    }

    // Get or generate query vectors
    let (queries, ground_truth): (Vec<Vec<f32>>, Option<Vec<Vec<u32>>>) = if let Some(ref ds) = dataset {
        let num_queries = args.num_queries.min(ds.num_queries());
        println!("Using {} query vectors from {} dataset", num_queries, ds.name);
        (
            ds.query_vectors.iter().take(num_queries).cloned().collect(),
            Some(ds.ground_truth.iter().take(num_queries).cloned().collect()),
        )
    } else {
        print!("Generating {} query vectors... ", args.num_queries);
        std::io::Write::flush(&mut std::io::stdout()).unwrap();
        let q = generate_random_vectors(args.num_queries, index.dimensions(), args.seed + 1000);
        println!("done");
        (q, None)
    };

    // Evaluate recall
    println!();
    println!("--- Evaluating Recall ---");
    let has_ground_truth = ground_truth.is_some();
    if has_ground_truth {
        println!("Using ground truth from dataset for recall computation");
    } else {
        println!("Using brute-force search for recall computation");
    }
    let start = Instant::now();

    let mut total_recall = 0.0;
    let mut search_times = Vec::with_capacity(queries.len());
    let mut total_probes_searched: usize = 0;

    for (i, query) in queries.iter().enumerate() {
        let search_start = Instant::now();
        let result = index.search(query, args.k);
        search_times.push(search_start.elapsed().as_secs_f64() * 1000.0); // ms
        total_probes_searched += result.probes_searched;

        let recall = if let Some(ref gt) = ground_truth {
            compute_recall_gt(&result.neighbors, &gt[i], args.k)
        } else {
            let exact = index.brute_force_search(query, args.k);
            compute_recall(&result.neighbors, &exact)
        };
        total_recall += recall;
    }

    let eval_time = start.elapsed();
    let avg_recall = total_recall / queries.len() as f64;
    let avg_probes_searched = total_probes_searched as f64 / queries.len() as f64;

    search_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50_latency = search_times[search_times.len() / 2];
    let p99_latency = search_times[(search_times.len() as f64 * 0.99) as usize];
    let avg_latency: f64 = search_times.iter().sum::<f64>() / search_times.len() as f64;

    println!("Evaluation complete in {:.2}s", eval_time.as_secs_f64());
    println!();
    println!("=== Results ===");
    println!("  Dataset:            {:>10}", args.dataset);
    println!("  Recall@{}:          {:>10.4}", args.k, avg_recall);
    println!("  Avg probes searched:{:>10.2}", avg_probes_searched);
    println!("  Avg latency:        {:>10.3} ms", avg_latency);
    println!("  P50 latency:        {:>10.3} ms", p50_latency);
    println!("  P99 latency:        {:>10.3} ms", p99_latency);
    println!("  QPS:                {:>10.1}", 1000.0 / avg_latency);

    // Compression analysis
    println!();
    println!("--- Compression Analysis ---");
    let comp_stats_info = index.stats();
    let num_samples = 100.min(comp_stats_info.num_centroids);
    print!("Analyzing {} random posting lists with zstd... ", num_samples);
    std::io::stdout().flush().unwrap();
    let comp_stats = index.analyze_compression(100, 3); // zstd level 3 (default)
    println!("done");
    println!();
    println!("=== Compression Results ===");
    println!("  Postings sampled:   {:>10}", comp_stats.num_postings);
    println!("  Vectors in sample:  {:>10}", comp_stats.total_vectors);
    println!("  Original size:      {:>10}", format_bytes(comp_stats.original_bytes));
    println!("  Compressed size:    {:>10}", format_bytes(comp_stats.compressed_bytes));
    println!("  Compression ratio:  {:>10.2}x", comp_stats.compression_ratio);
    println!("  Bytes/vector (orig):{:>10.1}",
        if comp_stats.total_vectors > 0 { comp_stats.original_bytes as f64 / comp_stats.total_vectors as f64 } else { 0.0 });
    println!("  Bytes/vector (comp):{:>10.1}",
        if comp_stats.total_vectors > 0 { comp_stats.compressed_bytes as f64 / comp_stats.total_vectors as f64 } else { 0.0 });
}
