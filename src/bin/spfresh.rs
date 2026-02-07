use clap::Parser;
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
    /// Next centroid ID
    next_centroid_id: u64,
    /// Total vectors in the index
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
            connectivity: 16,
            expansion_add: 128,
            expansion_search: 64,
            multi: false,
        };

        let centroid_index = new_index(&index_options).expect("Failed to create centroid index");
        centroid_index.reserve(1000).expect("Failed to reserve");

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

        // Find nearest centroid
        let nearest = self
            .centroid_index
            .search(vector, 1)
            .expect("Search failed");
        let centroid_id = nearest.keys[0];

        // Add to posting list
        self.posting_lists
            .entry(centroid_id)
            .or_default()
            .push(StoredVector {
                id,
                data: vector.to_vec(),
            });
        self.total_vectors += 1;

        // Check if split is needed
        if self.posting_lists[&centroid_id].len() > self.config.max_posting_size {
            self.split(centroid_id);
        }
    }

    fn add_to_centroids_index(&mut self, id: u64, v: &[f32]) {
        if let Err(e) = self.centroid_index.add(id, &v) {
            self.centroid_index.reserve(1024).expect("Failed to reserve");
            self.centroid_index.add(id, &v).expect("Failed to add to index");
        }
        self.centroid_vectors.insert(id, v.into());
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

        // Assign vectors to new centroids
        let mut list1 = Vec::new();
        let mut list2 = Vec::new();

        for v in vectors {
            let dist1 = Self::l2_distance(&v.data, &centroid1);
            let dist2 = Self::l2_distance(&v.data, &centroid2);
            if dist1 <= dist2 {
                list1.push(v);
            } else {
                list2.push(v);
            }
        }

        self.posting_lists.insert(new_id1, list1);
        self.posting_lists.insert(new_id2, list2);

        // Reassign vectors from neighboring centroids (SPFresh key insight)
        self.reassign_from_neighbors(new_id1);
        self.reassign_from_neighbors(new_id2);

        // Check if new centroids need merging (unlikely after split, but for consistency)
        //self.check_merge(new_id1);
        //self.check_merge(new_id2);
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

    /// Reassign vectors from neighboring centroids that might be closer to this centroid
    fn reassign_from_neighbors(&mut self, centroid_id: u64) {
        let centroid_vec = match self.centroid_vectors.get(&centroid_id) {
            Some(v) => v.clone(),
            None => return,
        };

        // Find neighboring centroids
        let neighbors = self
            .centroid_index
            .search(&centroid_vec, self.config.reassign_neighbors + 1)
            .expect("Search failed");

        let mut to_reassign: Vec<(u64, StoredVector)> = Vec::new();

        // Check each neighbor's posting list for vectors closer to this centroid
        for &neighbor_id in &neighbors.keys {
            if neighbor_id == centroid_id {
                continue;
            }

            let neighbor_vec = match self.centroid_vectors.get(&neighbor_id) {
                Some(v) => v.clone(),
                None => continue,
            };

            if let Some(posting_list) = self.posting_lists.get_mut(&neighbor_id) {
                let mut i = 0;
                while i < posting_list.len() {
                    let v = &posting_list[i];
                    let dist_to_current = Self::l2_distance(&v.data, &centroid_vec);
                    let dist_to_neighbor = Self::l2_distance(&v.data, &neighbor_vec);

                    if dist_to_current < dist_to_neighbor {
                        // This vector should be reassigned
                        let removed = posting_list.swap_remove(i);
                        to_reassign.push((centroid_id, removed));
                    } else {
                        i += 1;
                    }
                }
            }
        }

        // Add reassigned vectors to this centroid's posting list
        self.num_reassigned += to_reassign.len();
        for (target_id, vector) in to_reassign {
            self.posting_lists.entry(target_id).or_default().push(vector);
        }
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

    /// Merge a centroid with its nearest neighbor that won't cause an immediate split
    fn merge(&mut self, centroid_id: u64) {
        let centroid_vec = match self.centroid_vectors.get(&centroid_id) {
            Some(v) => v.clone(),
            None => return,
        };

        let my_size = self.posting_lists.get(&centroid_id).map(|l| l.len()).unwrap_or(0);

        // Find nearest neighbors and pick one where merging won't trigger immediate split
        let num_candidates = (self.centroid_vectors.len()).min(20); // Check up to 20 nearest
        let neighbors = self.centroid_index
            .search(&centroid_vec, num_candidates)
            .expect("Search failed");

        let mut neighbor_id = None;
        for &candidate_id in &neighbors.keys {
            if candidate_id == centroid_id {
                continue;
            }
            let candidate_size = self.posting_lists.get(&candidate_id).map(|l| l.len()).unwrap_or(0);
            // Only merge if combined size won't exceed max_posting_size
            if my_size + candidate_size <= self.config.max_posting_size {
                neighbor_id = Some(candidate_id);
                break;
            }
        }

        // If no suitable neighbor found, don't merge (avoid split-merge cycle)
        let neighbor_id = match neighbor_id {
            Some(id) => id,
            None => return,
        };

        self.num_merges += 1;

        // Remove this centroid
        let vectors = self.posting_lists.remove(&centroid_id).unwrap_or_default();
        let _ = self.centroid_index.remove(centroid_id); // Ignore error, soft-delete is fine
        self.centroid_vectors.remove(&centroid_id);

        // Get neighbor's existing vectors
        let mut merged_vectors = self.posting_lists.remove(&neighbor_id).unwrap_or_default();
        merged_vectors.extend(vectors);

        // Compute new centroid position
        let new_centroid = if !merged_vectors.is_empty() {
            self.compute_centroid(&merged_vectors)
        } else {
            return;
        };

        // Remove old neighbor from index (soft-delete)
        let _ = self.centroid_index.remove(neighbor_id);
        self.centroid_vectors.remove(&neighbor_id);

        // Create new centroid with fresh ID (usearch remove is soft-delete, can't reuse IDs)
        let new_id = self.next_centroid_id;
        self.next_centroid_id += 1;

        self.add_to_centroids_index(new_id, &new_centroid);
        self.posting_lists.insert(new_id, merged_vectors);

        // No need to check for split - we ensured merged size <= max_posting_size
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
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(u64, f32)> {
        assert_eq!(query.len(), self.config.dimensions, "Query dimension mismatch");

        // Find top centroids to probe
        let num_probes = self.config.num_probes.min(self.centroid_vectors.len());
        let centroid_results = self.centroid_index.search(query, num_probes).expect("Search failed");

        // Collect all candidate vectors from probed posting lists
        let mut candidates: Vec<(u64, f32)> = Vec::new();

        for &centroid_id in &centroid_results.keys {
            if let Some(posting_list) = self.posting_lists.get(&centroid_id) {
                for v in posting_list {
                    let dist = Self::l2_distance(query, &v.data);
                    candidates.push((v.id, dist));
                }
            }
        }

        // Sort by distance and return top k
        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        candidates.truncate(k);
        candidates
    }

    /// Brute force search (for computing recall)
    pub fn brute_force_search(&self, query: &[f32], k: usize) -> Vec<(u64, f32)> {
        let mut all_vectors: Vec<(u64, f32)> = Vec::new();

        for posting_list in self.posting_lists.values() {
            for v in posting_list {
                let dist = Self::l2_distance(query, &v.data);
                all_vectors.push((v.id, dist));
            }
        }

        all_vectors.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        all_vectors.truncate(k);
        all_vectors
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

    /// Set the number of probes (for tuning recall after loading)
    pub fn set_num_probes(&mut self, num_probes: usize) {
        self.config.num_probes = num_probes;
    }

    /// Get index statistics
    pub fn stats(&self) -> IndexStats {
        let posting_sizes: Vec<usize> = self.posting_lists.values().map(|l| l.len()).collect();
        let min_size = posting_sizes.iter().min().copied().unwrap_or(0);
        let max_size = posting_sizes.iter().max().copied().unwrap_or(0);
        let avg_size = if posting_sizes.is_empty() {
            0.0
        } else {
            posting_sizes.iter().sum::<usize>() as f64 / posting_sizes.len() as f64
        };

        IndexStats {
            num_centroids: self.centroid_vectors.len(),
            total_vectors: self.total_vectors,
            min_posting_size: min_size,
            max_posting_size: max_size,
            avg_posting_size: avg_size,
            num_splits: self.num_splits,
            num_merges: self.num_merges,
            num_reassigned: self.num_reassigned,
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
    fn save_vectors<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        // Write header: total_vectors, dimensions
        let total_vectors = self.total_vectors as u64;
        let dimensions = self.config.dimensions as u64;
        writer.write_all(&total_vectors.to_le_bytes())?;
        writer.write_all(&dimensions.to_le_bytes())?;

        // Write each vector: centroid_id (u64), vector_id (u64), vector (f32 * dimensions)
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
  "num_centroids": {},
  "total_vectors": {},
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
            stats.num_centroids,
            stats.total_vectors,
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
            connectivity: 16,
            expansion_add: 128,
            expansion_search: 64,
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

        Ok(Self {
            config,
            centroid_index,
            posting_lists,
            centroid_vectors,
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

        Ok(SPFreshConfig {
            dimensions: get_value("dimensions")?,
            max_posting_size: get_value("max_posting_size")?,
            min_posting_size: get_value("min_posting_size")?,
            num_probes: get_value("num_probes")?,
            reassign_neighbors: get_value("reassign_neighbors")?,
            kmeans_iters: get_value("kmeans_iters")?,
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

        // Read header
        let mut buf8 = [0u8; 8];
        reader.read_exact(&mut buf8)?;
        let total_vectors = u64::from_le_bytes(buf8) as usize;
        reader.read_exact(&mut buf8)?;
        let file_dimensions = u64::from_le_bytes(buf8) as usize;

        if file_dimensions != dimensions {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Dimension mismatch: expected {}, got {}", dimensions, file_dimensions),
            ));
        }

        let mut posting_lists: HashMap<u64, Vec<StoredVector>> = HashMap::new();
        let mut max_centroid_id: u64 = 0;
        let mut buf4 = [0u8; 4];

        for _ in 0..total_vectors {
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
    pub min_posting_size: usize,
    pub max_posting_size: usize,
    pub avg_posting_size: f64,
    pub num_splits: usize,
    pub num_merges: usize,
    pub num_reassigned: usize,
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

    /// Number of queries for recall evaluation
    #[arg(long, default_value_t = 1000)]
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

    let config = SPFreshConfig {
        dimensions: args.dimensions,
        max_posting_size: args.max_posting_size,
        min_posting_size: args.min_posting_size,
        num_probes: args.num_probes,
        reassign_neighbors: args.reassign_neighbors,
        kmeans_iters: args.kmeans_iters,
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

                // Override num_probes from command line if specified
                idx.set_num_probes(args.num_probes);

                let stats = idx.stats();
                println!();
                println!("Loaded index:");
                println!("  Num centroids:      {:>10}", stats.num_centroids);
                println!("  Total vectors:      {:>10}", stats.total_vectors);
                println!("  Min posting size:   {:>10}", stats.min_posting_size);
                println!("  Max posting size:   {:>10}", stats.max_posting_size);
                println!("  Avg posting size:   {:>10.1}", stats.avg_posting_size);
                println!("  Num probes:         {:>10} (from args)", args.num_probes);
                idx
            }
            Err(e) => {
                println!("FAILED: {}", e);
                std::process::exit(1);
            }
        }
    } else {
        // Build new index
        println!("Configuration:");
        println!("  Vectors:            {:>10}", args.num_vectors);
        println!("  Dimensions:         {:>10}", args.dimensions);
        println!("  Max posting size:   {:>10}", args.max_posting_size);
        println!("  Min posting size:   {:>10}", args.min_posting_size);
        println!("  Num probes:         {:>10}", args.num_probes);
        println!("  Reassign neighbors: {:>10}", args.reassign_neighbors);
        println!("  K-means iters:      {:>10}", args.kmeans_iters);
        println!("  Num queries:        {:>10}", args.num_queries);
        println!("  K (for k-NN):       {:>10}", args.k);
        println!("  Seed:               {:>10}", args.seed);
        println!();

        // Generate data
        print!("Generating {} vectors... ", args.num_vectors);
        std::io::Write::flush(&mut std::io::stdout()).unwrap();
        let start = Instant::now();
        let vectors = generate_random_vectors(args.num_vectors, args.dimensions, args.seed);
        println!("done ({:.2}s)", start.elapsed().as_secs_f64());

        // Build index
        println!();
        println!("--- Building Index ---");
        let mut idx = SPFreshIndex::new(config);
        let start = Instant::now();

        let report_interval = (args.num_vectors / 20).max(1000).min(args.num_vectors);
        let mut last_report = Instant::now();

        for (i, vector) in vectors.iter().enumerate() {
            idx.insert(i as u64, vector);

            // Report every interval or every 10 seconds, whichever comes first
            let should_report = (i + 1) % report_interval == 0
                || last_report.elapsed().as_secs_f64() >= 1.0
                || i + 1 == args.num_vectors;

            if should_report && i > 0 {
                let elapsed = start.elapsed().as_secs_f64();
                let rate = (i + 1) as f64 / elapsed;
                let stats = idx.stats();

                // Use carriage return to overwrite the line
                println!(
                    "{:>6.1}% | {:>8} vecs | {:>5} centroids | {:>5} splits | {:>5} merges | {:>7} reassigned | {:>8.0} vec/s | {:.1}s",
                    (i + 1) as f64 / args.num_vectors as f64 * 100.0,
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
        println!("Build complete in {:.2}s ({:.0} vectors/sec)", build_time.as_secs_f64(), args.num_vectors as f64 / build_time.as_secs_f64());
        println!("  Num centroids:      {:>10}", stats.num_centroids);
        println!("  Total vectors:      {:>10}", stats.total_vectors);
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

    // Generate query vectors (using index dimensions for compatibility with loaded indices)
    print!("Generating {} query vectors... ", args.num_queries);
    std::io::Write::flush(&mut std::io::stdout()).unwrap();
    let queries = generate_random_vectors(args.num_queries, index.dimensions(), args.seed + 1000);
    println!("done");

    // Evaluate recall
    println!();
    println!("--- Evaluating Recall ---");
    let start = Instant::now();

    let mut total_recall = 0.0;
    let mut search_times = Vec::with_capacity(args.num_queries);

    for query in &queries {
        let search_start = Instant::now();
        let approximate = index.search(query, args.k);
        search_times.push(search_start.elapsed().as_secs_f64() * 1000.0); // ms

        let exact = index.brute_force_search(query, args.k);
        let recall = compute_recall(&approximate, &exact);
        println!("{:?}: query recall: {}", Instant::now(), recall);
        total_recall += recall;
    }

    let eval_time = start.elapsed();
    let avg_recall = total_recall / args.num_queries as f64;

    search_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50_latency = search_times[search_times.len() / 2];
    let p99_latency = search_times[(search_times.len() as f64 * 0.99) as usize];
    let avg_latency: f64 = search_times.iter().sum::<f64>() / search_times.len() as f64;

    println!("Evaluation complete in {:.2}s", eval_time.as_secs_f64());
    println!();
    println!("=== Results ===");
    println!("  Recall@{}:          {:>10.4}", args.k, avg_recall);
    println!("  Avg latency:        {:>10.3} ms", avg_latency);
    println!("  P50 latency:        {:>10.3} ms", p50_latency);
    println!("  P99 latency:        {:>10.3} ms", p99_latency);
    println!("  QPS:                {:>10.1}", 1000.0 / avg_latency);
}
