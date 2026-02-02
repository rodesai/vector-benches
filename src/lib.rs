use rand::Rng;
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use usearch::{new_index, Index, IndexOptions, MetricKind, ScalarKind};

/// Default configuration for benchmarks
pub const DEFAULT_NUM_VECTORS: usize = 1_000_000;
pub const DEFAULT_DIMENSIONS: usize = 128;
pub const DEFAULT_CONNECTIVITY: usize = 16;
pub const DEFAULT_EXPANSION_ADD: usize = 128;
pub const DEFAULT_EXPANSION_SEARCH: usize = 64;

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
        }
    }
}

impl BenchConfig {
    pub fn with_num_vectors(mut self, n: usize) -> Self {
        self.num_vectors = n;
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

/// Build an index from vectors
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
