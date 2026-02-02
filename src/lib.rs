use rand::Rng;
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;
use usearch::{new_index, Index, IndexOptions, MetricKind, ScalarKind};

/// Default configuration for benchmarks
pub const DEFAULT_NUM_VECTORS: usize = 1_000_000;
pub const DEFAULT_DIMENSIONS: usize = 128;
pub const DEFAULT_CONNECTIVITY: usize = 16;
pub const DEFAULT_EXPANSION_ADD: usize = 128;
pub const DEFAULT_EXPANSION_SEARCH: usize = 64;
pub const DEFAULT_NUM_THREADS: usize = 0; // 0 means use all available cores

/// Configuration for benchmark runs
#[derive(Clone)]
pub struct BenchConfig {
    pub num_vectors: usize,
    pub dimensions: usize,
    pub connectivity: usize,
    pub expansion_add: usize,
    pub expansion_search: usize,
    pub metric: MetricKind,
    pub quantization: ScalarKind,
    pub num_threads: usize,
}

impl Default for BenchConfig {
    fn default() -> Self {
        Self {
            num_vectors: DEFAULT_NUM_VECTORS,
            dimensions: DEFAULT_DIMENSIONS,
            connectivity: DEFAULT_CONNECTIVITY,
            expansion_add: DEFAULT_EXPANSION_ADD,
            expansion_search: DEFAULT_EXPANSION_SEARCH,
            metric: MetricKind::Cos,
            quantization: ScalarKind::F32,
            num_threads: DEFAULT_NUM_THREADS,
        }
    }
}

impl BenchConfig {
    /// Returns the effective number of threads (resolves 0 to available parallelism)
    pub fn effective_threads(&self) -> usize {
        if self.num_threads == 0 {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
        } else {
            self.num_threads
        }
    }
}

impl BenchConfig {
    pub fn with_num_vectors(mut self, n: usize) -> Self {
        self.num_vectors = n;
        self
    }

    pub fn with_num_threads(mut self, n: usize) -> Self {
        self.num_threads = n;
        self
    }

    pub fn index_options(&self) -> IndexOptions {
        IndexOptions {
            dimensions: self.dimensions,
            metric: self.metric,
            quantization: self.quantization,
            connectivity: self.connectivity,
            expansion_add: self.expansion_add,
            expansion_search: self.expansion_search,
            multi: false,
        }
    }
}

/// Generate random vectors for benchmarking
pub fn generate_random_vectors(num_vectors: usize, dimensions: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    (0..num_vectors)
        .map(|_| (0..dimensions).map(|_| rng.gen::<f32>()).collect())
        .collect()
}

/// Partitioned vectors - each thread gets its own buffer with (start_key, vectors)
pub struct PartitionedVectors {
    /// Each partition contains (starting_key, vectors)
    pub partitions: Vec<(u64, Vec<Vec<f32>>)>,
}

/// Generate random vectors partitioned for multi-threaded building.
/// Each thread gets its own Vec to avoid any locking during index construction.
pub fn generate_partitioned_vectors(
    num_vectors: usize,
    dimensions: usize,
    num_threads: usize,
    seed: u64,
) -> PartitionedVectors {
    let vectors_per_thread = num_vectors / num_threads;
    let remainder = num_vectors % num_threads;

    // Generate partitions in parallel, each with its own RNG seeded deterministically
    let partitions: Vec<(u64, Vec<Vec<f32>>)> = (0..num_threads)
        .into_par_iter()
        .map(|thread_idx| {
            // Calculate this thread's range
            let start_idx = thread_idx * vectors_per_thread + thread_idx.min(remainder);
            let count = vectors_per_thread + if thread_idx < remainder { 1 } else { 0 };

            // Each thread gets a deterministic seed based on the base seed and thread index
            let thread_seed = seed.wrapping_add(thread_idx as u64 * 0x9E3779B97F4A7C15);
            let mut rng = Xoshiro256PlusPlus::seed_from_u64(thread_seed);

            let vectors: Vec<Vec<f32>> = (0..count)
                .map(|_| (0..dimensions).map(|_| rng.gen::<f32>()).collect())
                .collect();

            (start_idx as u64, vectors)
        })
        .collect();

    PartitionedVectors { partitions }
}

/// Build an index from vectors (single-threaded)
pub fn build_index(config: &BenchConfig, vectors: &[Vec<f32>]) -> Index {
    let index = new_index(&config.index_options()).expect("Failed to create index");
    index
        .reserve(vectors.len())
        .expect("Failed to reserve capacity");

    for (i, vector) in vectors.iter().enumerate() {
        index.add(i as u64, vector).expect("Failed to add vector");
    }

    index
}

/// Build an index from partitioned vectors using multiple threads.
/// Each thread reads from its own partition buffer, avoiding any locking on the data side.
/// The usearch Index itself is thread-safe for concurrent insertions.
pub fn build_index_parallel(config: &BenchConfig, partitioned: &PartitionedVectors) -> Index {
    let total_vectors: usize = partitioned.partitions.iter().map(|(_, v)| v.len()).sum();

    let index = new_index(&config.index_options()).expect("Failed to create index");
    index
        .reserve(total_vectors)
        .expect("Failed to reserve capacity");

    // Use rayon to process partitions in parallel
    // Each thread works on its own partition, no locking needed on the vector data
    partitioned.partitions.par_iter().for_each(|(start_key, vectors)| {
        for (offset, vector) in vectors.iter().enumerate() {
            let key = start_key + offset as u64;
            index.add(key, vector).expect("Failed to add vector");
        }
    });

    index
}

/// Serialize index to a buffer
pub fn serialize_to_buffer(index: &Index) -> Vec<u8> {
    let size = index.serialized_length();
    let mut buffer = vec![0u8; size];
    index
        .save_to_buffer(&mut buffer)
        .expect("Failed to serialize index");
    buffer
}

/// Load index from a buffer
pub fn load_from_buffer(config: &BenchConfig, buffer: &[u8]) -> Index {
    let index = new_index(&config.index_options()).expect("Failed to create index");
    index
        .load_from_buffer(buffer)
        .expect("Failed to load index from buffer");
    index
}
