//! Dataset loading module for ANN benchmark datasets
//!
//! Supports loading vectors and ground truth from standard benchmark datasets:
//! - SIFT1M: 128-dim vectors from http://corpus-texmex.irisa.fr/
//! - SIFT100M: 100M subset of BigANN/ANN_SIFT1B (128-dim, bvecs format)
//! - DEEP1M: 96-dim vectors from Yandex DEEP1B (first 1M)
//!
//! File formats:
//! - fvecs: float32 vectors with dimension prefix per vector
//! - ivecs: int32 vectors (used for ground truth indices)
//! - bvecs: uint8 vectors with dimension prefix per vector

use std::fs::File;
use std::io::{BufReader, Read, Result, Error, ErrorKind};
use std::path::Path;

/// A loaded dataset with base vectors, query vectors, and ground truth
pub struct Dataset {
    /// Name of the dataset
    pub name: String,
    /// Base vectors to index
    pub base_vectors: Vec<Vec<f32>>,
    /// Query vectors for evaluation
    pub query_vectors: Vec<Vec<f32>>,
    /// Ground truth: for each query, the indices of k nearest neighbors in base_vectors
    pub ground_truth: Vec<Vec<u32>>,
    /// Number of dimensions
    pub dimensions: usize,
}

impl Dataset {
    /// Get the number of base vectors
    pub fn num_base(&self) -> usize {
        self.base_vectors.len()
    }

    /// Get the number of query vectors
    pub fn num_queries(&self) -> usize {
        self.query_vectors.len()
    }

    /// Get the number of ground truth neighbors per query
    pub fn ground_truth_k(&self) -> usize {
        self.ground_truth.first().map(|v| v.len()).unwrap_or(0)
    }
}

/// Dataset type enum for CLI
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DatasetType {
    Random,
    Sift1M,
    Sift100M,
    Deep1M,
}

impl std::str::FromStr for DatasetType {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "random" => Ok(DatasetType::Random),
            "sift1m" | "sift" => Ok(DatasetType::Sift1M),
            "sift100m" | "bigann" => Ok(DatasetType::Sift100M),
            "deep1m" | "deep" => Ok(DatasetType::Deep1M),
            _ => Err(format!("Unknown dataset type: {}. Valid options: random, sift1m, sift100m, deep1m", s)),
        }
    }
}

impl std::fmt::Display for DatasetType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DatasetType::Random => write!(f, "random"),
            DatasetType::Sift1M => write!(f, "sift1m"),
            DatasetType::Sift100M => write!(f, "sift100m"),
            DatasetType::Deep1M => write!(f, "deep1m"),
        }
    }
}

/// Dataset info with download URLs and file structure
pub struct DatasetInfo {
    pub name: &'static str,
    pub dimensions: usize,
    pub base_url: &'static str,
    pub archive_name: &'static str,
    pub base_file: &'static str,
    pub query_file: &'static str,
    pub groundtruth_file: &'static str,
    pub num_base: usize,
    pub num_queries: usize,
}

impl DatasetInfo {
    pub fn sift1m() -> Self {
        Self {
            name: "SIFT1M",
            dimensions: 128,
            base_url: "ftp://ftp.irisa.fr/local/texmex/corpus/",
            archive_name: "sift.tar.gz",
            base_file: "sift/sift_base.fvecs",
            query_file: "sift/sift_query.fvecs",
            groundtruth_file: "sift/sift_groundtruth.ivecs",
            num_base: 1_000_000,
            num_queries: 10_000,
        }
    }

    pub fn deep1m() -> Self {
        Self {
            name: "DEEP1M",
            dimensions: 96,
            // DEEP1B dataset from Yandex - need to use a subset
            base_url: "https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/",
            archive_name: "base.1M.fbin",  // Binary format, 1M subset
            base_file: "deep1m_base.fvecs",
            query_file: "deep1m_query.fvecs",
            groundtruth_file: "deep1m_groundtruth.ivecs",
            num_base: 1_000_000,
            num_queries: 10_000,
        }
    }

    pub fn sift100m() -> Self {
        Self {
            name: "SIFT100M",
            dimensions: 128,
            // BigANN / ANN_SIFT1B dataset - first 100M vectors
            base_url: "ftp://ftp.irisa.fr/local/texmex/corpus/",
            archive_name: "bigann_base.bvecs.gz",
            base_file: "bigann_base.bvecs",
            query_file: "bigann_query.bvecs",
            groundtruth_file: "gnd/idx_100M.ivecs",
            num_base: 100_000_000,
            num_queries: 10_000,
        }
    }
}

/// Read vectors from an fvecs file (float32 vectors)
/// Format: for each vector, 4 bytes dimension (int32), then dim*4 bytes of float32 values
pub fn read_fvecs<P: AsRef<Path>>(path: P) -> Result<Vec<Vec<f32>>> {
    let file = File::open(path.as_ref())?;
    let file_size = file.metadata()?.len();
    let mut reader = BufReader::new(file);

    let mut vectors = Vec::new();
    let mut buf4 = [0u8; 4];
    let mut bytes_read: u64 = 0;

    while bytes_read < file_size {
        // Read dimension (int32, little-endian)
        if reader.read_exact(&mut buf4).is_err() {
            break;
        }
        bytes_read += 4;
        let dim = i32::from_le_bytes(buf4) as usize;

        if dim == 0 || dim > 10000 {
            return Err(Error::new(
                ErrorKind::InvalidData,
                format!("Invalid dimension {} in fvecs file", dim),
            ));
        }

        // Read vector data
        let mut vector = Vec::with_capacity(dim);
        for _ in 0..dim {
            reader.read_exact(&mut buf4)?;
            bytes_read += 4;
            vector.push(f32::from_le_bytes(buf4));
        }
        vectors.push(vector);
    }

    Ok(vectors)
}

/// Read vectors from an fvecs file with a limit on number of vectors
pub fn read_fvecs_limited<P: AsRef<Path>>(path: P, max_vectors: usize) -> Result<Vec<Vec<f32>>> {
    let file = File::open(path.as_ref())?;
    let mut reader = BufReader::new(file);

    let mut vectors = Vec::new();
    let mut buf4 = [0u8; 4];

    while vectors.len() < max_vectors {
        // Read dimension (int32, little-endian)
        if reader.read_exact(&mut buf4).is_err() {
            break;
        }
        let dim = i32::from_le_bytes(buf4) as usize;

        if dim == 0 || dim > 10000 {
            return Err(Error::new(
                ErrorKind::InvalidData,
                format!("Invalid dimension {} in fvecs file", dim),
            ));
        }

        // Read vector data
        let mut vector = Vec::with_capacity(dim);
        for _ in 0..dim {
            reader.read_exact(&mut buf4)?;
            vector.push(f32::from_le_bytes(buf4));
        }
        vectors.push(vector);
    }

    Ok(vectors)
}

/// Read vectors from an ivecs file (int32 vectors, used for ground truth)
/// Format: for each vector, 4 bytes dimension (int32), then dim*4 bytes of int32 values
pub fn read_ivecs<P: AsRef<Path>>(path: P) -> Result<Vec<Vec<u32>>> {
    let file = File::open(path.as_ref())?;
    let file_size = file.metadata()?.len();
    let mut reader = BufReader::new(file);

    let mut vectors = Vec::new();
    let mut buf4 = [0u8; 4];
    let mut bytes_read: u64 = 0;

    while bytes_read < file_size {
        // Read dimension (int32, little-endian)
        if reader.read_exact(&mut buf4).is_err() {
            break;
        }
        bytes_read += 4;
        let dim = i32::from_le_bytes(buf4) as usize;

        if dim == 0 || dim > 10000 {
            return Err(Error::new(
                ErrorKind::InvalidData,
                format!("Invalid dimension {} in ivecs file", dim),
            ));
        }

        // Read vector data (as u32 for indices)
        let mut vector = Vec::with_capacity(dim);
        for _ in 0..dim {
            reader.read_exact(&mut buf4)?;
            bytes_read += 4;
            vector.push(u32::from_le_bytes(buf4));
        }
        vectors.push(vector);
    }

    Ok(vectors)
}

/// Read vectors from a bvecs file (uint8 vectors)
/// Format: for each vector, 4 bytes dimension (int32), then dim bytes of uint8 values
/// Converts to f32 for compatibility
pub fn read_bvecs<P: AsRef<Path>>(path: P) -> Result<Vec<Vec<f32>>> {
    let file = File::open(path.as_ref())?;
    let file_size = file.metadata()?.len();
    let mut reader = BufReader::new(file);

    let mut vectors = Vec::new();
    let mut buf4 = [0u8; 4];
    let mut bytes_read: u64 = 0;

    while bytes_read < file_size {
        // Read dimension (int32, little-endian)
        if reader.read_exact(&mut buf4).is_err() {
            break;
        }
        bytes_read += 4;
        let dim = i32::from_le_bytes(buf4) as usize;

        if dim == 0 || dim > 10000 {
            return Err(Error::new(
                ErrorKind::InvalidData,
                format!("Invalid dimension {} in bvecs file", dim),
            ));
        }

        // Read vector data (uint8, convert to f32)
        let mut vector = Vec::with_capacity(dim);
        let mut buf1 = [0u8; 1];
        for _ in 0..dim {
            reader.read_exact(&mut buf1)?;
            bytes_read += 1;
            vector.push(buf1[0] as f32);
        }
        vectors.push(vector);
    }

    Ok(vectors)
}

/// Read vectors from a bvecs file with a limit
pub fn read_bvecs_limited<P: AsRef<Path>>(path: P, max_vectors: usize) -> Result<Vec<Vec<f32>>> {
    let file = File::open(path.as_ref())?;
    let mut reader = BufReader::new(file);

    let mut vectors = Vec::new();
    let mut buf4 = [0u8; 4];

    while vectors.len() < max_vectors {
        // Read dimension
        if reader.read_exact(&mut buf4).is_err() {
            break;
        }
        let dim = i32::from_le_bytes(buf4) as usize;

        if dim == 0 || dim > 10000 {
            return Err(Error::new(
                ErrorKind::InvalidData,
                format!("Invalid dimension {} in bvecs file", dim),
            ));
        }

        // Read vector data (uint8, convert to f32)
        let mut vector = Vec::with_capacity(dim);
        let mut buf1 = [0u8; 1];
        for _ in 0..dim {
            reader.read_exact(&mut buf1)?;
            vector.push(buf1[0] as f32);
        }
        vectors.push(vector);
    }

    Ok(vectors)
}

/// Read vectors from a fbin file (binary float format used by some datasets)
/// Format: 4 bytes num_vectors (uint32), 4 bytes dimension (uint32), then vectors in row-major order
pub fn read_fbin<P: AsRef<Path>>(path: P) -> Result<Vec<Vec<f32>>> {
    let file = File::open(path.as_ref())?;
    let mut reader = BufReader::new(file);

    let mut buf4 = [0u8; 4];

    // Read header
    reader.read_exact(&mut buf4)?;
    let num_vectors = u32::from_le_bytes(buf4) as usize;

    reader.read_exact(&mut buf4)?;
    let dim = u32::from_le_bytes(buf4) as usize;

    // Read vectors
    let mut vectors = Vec::with_capacity(num_vectors);
    for _ in 0..num_vectors {
        let mut vector = Vec::with_capacity(dim);
        for _ in 0..dim {
            reader.read_exact(&mut buf4)?;
            vector.push(f32::from_le_bytes(buf4));
        }
        vectors.push(vector);
    }

    Ok(vectors)
}

/// Read vectors from a fbin file with a limit
pub fn read_fbin_limited<P: AsRef<Path>>(path: P, max_vectors: usize) -> Result<Vec<Vec<f32>>> {
    let file = File::open(path.as_ref())?;
    let mut reader = BufReader::new(file);

    let mut buf4 = [0u8; 4];

    // Read header
    reader.read_exact(&mut buf4)?;
    let num_vectors = u32::from_le_bytes(buf4) as usize;

    reader.read_exact(&mut buf4)?;
    let dim = u32::from_le_bytes(buf4) as usize;

    let to_read = num_vectors.min(max_vectors);

    // Read vectors
    let mut vectors = Vec::with_capacity(to_read);
    for _ in 0..to_read {
        let mut vector = Vec::with_capacity(dim);
        for _ in 0..dim {
            reader.read_exact(&mut buf4)?;
            vector.push(f32::from_le_bytes(buf4));
        }
        vectors.push(vector);
    }

    Ok(vectors)
}

/// Load SIFT1M dataset from a directory
/// Expected files:
///   sift_base.fvecs or sift/sift_base.fvecs (1M vectors, 128-dim)
///   sift_query.fvecs or sift/sift_query.fvecs (10K vectors)
///   sift_groundtruth.ivecs or sift/sift_groundtruth.ivecs (10K x 100 indices)
pub fn load_sift1m<P: AsRef<Path>>(dir: P, max_base_vectors: Option<usize>) -> Result<Dataset> {
    let dir = dir.as_ref();

    // Try both flat and nested directory structures
    let (base_path, query_path, gt_path) = if dir.join("sift_base.fvecs").exists() {
        (
            dir.join("sift_base.fvecs"),
            dir.join("sift_query.fvecs"),
            dir.join("sift_groundtruth.ivecs"),
        )
    } else if dir.join("sift/sift_base.fvecs").exists() {
        (
            dir.join("sift/sift_base.fvecs"),
            dir.join("sift/sift_query.fvecs"),
            dir.join("sift/sift_groundtruth.ivecs"),
        )
    } else {
        return Err(Error::new(
            ErrorKind::NotFound,
            format!(
                "SIFT1M files not found in {}. Expected sift_base.fvecs, sift_query.fvecs, sift_groundtruth.ivecs",
                dir.display()
            ),
        ));
    };

    println!("Loading SIFT1M dataset from {}...", dir.display());

    // Load base vectors
    print!("  Loading base vectors... ");
    std::io::Write::flush(&mut std::io::stdout()).ok();
    let base_vectors = match max_base_vectors {
        Some(max) => read_fvecs_limited(&base_path, max)?,
        None => read_fvecs(&base_path)?,
    };
    println!("{} vectors, {} dimensions", base_vectors.len(), base_vectors.first().map(|v| v.len()).unwrap_or(0));

    // Load query vectors
    print!("  Loading query vectors... ");
    std::io::Write::flush(&mut std::io::stdout()).ok();
    let query_vectors = read_fvecs(&query_path)?;
    println!("{} vectors", query_vectors.len());

    // Load ground truth
    print!("  Loading ground truth... ");
    std::io::Write::flush(&mut std::io::stdout()).ok();
    let ground_truth = read_ivecs(&gt_path)?;
    println!("{} queries, k={}", ground_truth.len(), ground_truth.first().map(|v| v.len()).unwrap_or(0));

    let dimensions = base_vectors.first().map(|v| v.len()).unwrap_or(128);

    Ok(Dataset {
        name: "SIFT1M".to_string(),
        base_vectors,
        query_vectors,
        ground_truth,
        dimensions,
    })
}

/// Load DEEP1M dataset from a directory
/// Expected files (fvecs format):
///   deep1m_base.fvecs (1M vectors, 96-dim)
///   deep1m_query.fvecs (10K vectors)
///   deep1m_groundtruth.ivecs (10K x 100 indices)
/// Or fbin format:
///   base.1M.fbin
///   query.public.10K.fbin
///   groundtruth.public.10K.ibin
pub fn load_deep1m<P: AsRef<Path>>(dir: P, max_base_vectors: Option<usize>) -> Result<Dataset> {
    let dir = dir.as_ref();

    // Try fvecs format first
    if dir.join("deep1m_base.fvecs").exists() {
        println!("Loading DEEP1M dataset from {} (fvecs format)...", dir.display());

        print!("  Loading base vectors... ");
        std::io::Write::flush(&mut std::io::stdout()).ok();
        let base_vectors = match max_base_vectors {
            Some(max) => read_fvecs_limited(dir.join("deep1m_base.fvecs"), max)?,
            None => read_fvecs(dir.join("deep1m_base.fvecs"))?,
        };
        println!("{} vectors, {} dimensions", base_vectors.len(), base_vectors.first().map(|v| v.len()).unwrap_or(0));

        print!("  Loading query vectors... ");
        std::io::Write::flush(&mut std::io::stdout()).ok();
        let query_vectors = read_fvecs(dir.join("deep1m_query.fvecs"))?;
        println!("{} vectors", query_vectors.len());

        print!("  Loading ground truth... ");
        std::io::Write::flush(&mut std::io::stdout()).ok();
        let ground_truth = read_ivecs(dir.join("deep1m_groundtruth.ivecs"))?;
        println!("{} queries, k={}", ground_truth.len(), ground_truth.first().map(|v| v.len()).unwrap_or(0));

        let dimensions = base_vectors.first().map(|v| v.len()).unwrap_or(96);

        return Ok(Dataset {
            name: "DEEP1M".to_string(),
            base_vectors,
            query_vectors,
            ground_truth,
            dimensions,
        });
    }

    // Try fbin format (from Yandex)
    if dir.join("base.1M.fbin").exists() {
        println!("Loading DEEP1M dataset from {} (fbin format)...", dir.display());

        print!("  Loading base vectors... ");
        std::io::Write::flush(&mut std::io::stdout()).ok();
        let base_vectors = match max_base_vectors {
            Some(max) => read_fbin_limited(dir.join("base.1M.fbin"), max)?,
            None => read_fbin(dir.join("base.1M.fbin"))?,
        };
        println!("{} vectors, {} dimensions", base_vectors.len(), base_vectors.first().map(|v| v.len()).unwrap_or(0));

        // Query and groundtruth might be in different formats
        let query_path = if dir.join("query.public.10K.fbin").exists() {
            dir.join("query.public.10K.fbin")
        } else {
            dir.join("deep1m_query.fvecs")
        };

        print!("  Loading query vectors... ");
        std::io::Write::flush(&mut std::io::stdout()).ok();
        let query_vectors = if query_path.extension().map(|e| e == "fbin").unwrap_or(false) {
            read_fbin(&query_path)?
        } else {
            read_fvecs(&query_path)?
        };
        println!("{} vectors", query_vectors.len());

        // Ground truth
        let gt_path = if dir.join("groundtruth.public.10K.ibin").exists() {
            dir.join("groundtruth.public.10K.ibin")
        } else {
            dir.join("deep1m_groundtruth.ivecs")
        };

        print!("  Loading ground truth... ");
        std::io::Write::flush(&mut std::io::stdout()).ok();
        let ground_truth = read_ivecs(&gt_path)?;
        println!("{} queries, k={}", ground_truth.len(), ground_truth.first().map(|v| v.len()).unwrap_or(0));

        let dimensions = base_vectors.first().map(|v| v.len()).unwrap_or(96);

        return Ok(Dataset {
            name: "DEEP1M".to_string(),
            base_vectors,
            query_vectors,
            ground_truth,
            dimensions,
        });
    }

    Err(Error::new(
        ErrorKind::NotFound,
        format!(
            "DEEP1M files not found in {}. Expected either:\n  \
             - deep1m_base.fvecs, deep1m_query.fvecs, deep1m_groundtruth.ivecs\n  \
             - base.1M.fbin, query.public.10K.fbin, groundtruth.public.10K.ibin",
            dir.display()
        ),
    ))
}

/// Load SIFT100M dataset (BigANN / ANN_SIFT1B subset) from a directory
/// Expected files:
///   bigann_base.bvecs (1B vectors, but we read up to max_base_vectors)
///   bigann_query.bvecs (10K query vectors, 128-dim)
///   gnd/idx_100M.ivecs (ground truth for 100M subset)
///
/// Note: The base file is in bvecs format (uint8), which we convert to f32.
/// Ground truth files are named by the subset size (idx_1M.ivecs, idx_10M.ivecs, idx_100M.ivecs, etc.)
pub fn load_sift100m<P: AsRef<Path>>(dir: P, max_base_vectors: Option<usize>) -> Result<Dataset> {
    let dir = dir.as_ref();

    // Find base file
    let base_path = if dir.join("bigann_base.bvecs").exists() {
        dir.join("bigann_base.bvecs")
    } else if dir.join("ANN_SIFT1B/bigann_base.bvecs").exists() {
        dir.join("ANN_SIFT1B/bigann_base.bvecs")
    } else {
        return Err(Error::new(
            ErrorKind::NotFound,
            format!(
                "SIFT100M/BigANN base file not found in {}. Expected bigann_base.bvecs",
                dir.display()
            ),
        ));
    };

    // Find query file
    let query_path = if dir.join("bigann_query.bvecs").exists() {
        dir.join("bigann_query.bvecs")
    } else if dir.join("ANN_SIFT1B/bigann_query.bvecs").exists() {
        dir.join("ANN_SIFT1B/bigann_query.bvecs")
    } else {
        return Err(Error::new(
            ErrorKind::NotFound,
            format!(
                "SIFT100M/BigANN query file not found in {}. Expected bigann_query.bvecs",
                dir.display()
            ),
        ));
    };

    // Find ground truth file - try different sizes based on max_base_vectors
    let max_vecs = max_base_vectors.unwrap_or(100_000_000);
    let gt_path = find_bigann_groundtruth(dir, max_vecs)?;

    println!("Loading SIFT100M/BigANN dataset from {}...", dir.display());

    // Load base vectors (bvecs format - uint8 converted to f32)
    print!("  Loading base vectors (bvecs)... ");
    std::io::Write::flush(&mut std::io::stdout()).ok();
    let base_vectors = match max_base_vectors {
        Some(max) => read_bvecs_limited(&base_path, max)?,
        None => read_bvecs_limited(&base_path, 100_000_000)?, // Default to 100M
    };
    println!("{} vectors, {} dimensions", base_vectors.len(), base_vectors.first().map(|v| v.len()).unwrap_or(0));

    // Load query vectors (also bvecs format)
    print!("  Loading query vectors (bvecs)... ");
    std::io::Write::flush(&mut std::io::stdout()).ok();
    let query_vectors = read_bvecs(&query_path)?;
    println!("{} vectors", query_vectors.len());

    // Load ground truth
    print!("  Loading ground truth from {}... ", gt_path.file_name().unwrap_or_default().to_string_lossy());
    std::io::Write::flush(&mut std::io::stdout()).ok();
    let ground_truth = read_ivecs(&gt_path)?;
    println!("{} queries, k={}", ground_truth.len(), ground_truth.first().map(|v| v.len()).unwrap_or(0));

    let dimensions = base_vectors.first().map(|v| v.len()).unwrap_or(128);

    Ok(Dataset {
        name: format!("SIFT{}M", base_vectors.len() / 1_000_000),
        base_vectors,
        query_vectors,
        ground_truth,
        dimensions,
    })
}

/// Find the appropriate ground truth file for BigANN based on the number of base vectors
fn find_bigann_groundtruth<P: AsRef<Path>>(dir: P, num_vectors: usize) -> Result<std::path::PathBuf> {
    let dir = dir.as_ref();

    // Ground truth files are named by subset size
    let sizes = [
        (1_000_000, "idx_1M.ivecs"),
        (10_000_000, "idx_10M.ivecs"),
        (100_000_000, "idx_100M.ivecs"),
        (1_000_000_000, "idx_1000M.ivecs"),
    ];

    // Find the smallest ground truth file that covers our data
    let gt_name = sizes
        .iter()
        .find(|(size, _)| *size >= num_vectors)
        .map(|(_, name)| *name)
        .unwrap_or("idx_100M.ivecs");

    // Try different directory structures
    let candidates = [
        dir.join("gnd").join(gt_name),
        dir.join(gt_name),
        dir.join("ANN_SIFT1B/gnd").join(gt_name),
    ];

    for path in &candidates {
        if path.exists() {
            return Ok(path.clone());
        }
    }

    // If we can't find the exact file, try to find any ground truth file
    for (_, name) in sizes.iter().rev() {
        let candidates = [
            dir.join("gnd").join(name),
            dir.join(name),
            dir.join("ANN_SIFT1B/gnd").join(name),
        ];
        for path in &candidates {
            if path.exists() {
                return Ok(path.clone());
            }
        }
    }

    Err(Error::new(
        ErrorKind::NotFound,
        format!(
            "Ground truth file not found in {}. Expected gnd/idx_100M.ivecs or similar.\n\
             Download from: ftp://ftp.irisa.fr/local/texmex/corpus/bigann_gnd.tar.gz",
            dir.display()
        ),
    ))
}

/// Load a dataset by type from a directory
pub fn load_dataset(dataset_type: DatasetType, dir: &Path, max_vectors: Option<usize>) -> Result<Dataset> {
    match dataset_type {
        DatasetType::Random => Err(Error::new(
            ErrorKind::InvalidInput,
            "Random dataset should be generated, not loaded from directory",
        )),
        DatasetType::Sift1M => load_sift1m(dir, max_vectors),
        DatasetType::Sift100M => load_sift100m(dir, max_vectors),
        DatasetType::Deep1M => load_deep1m(dir, max_vectors),
    }
}

/// Print dataset download instructions
pub fn print_download_instructions(dataset_type: DatasetType) {
    match dataset_type {
        DatasetType::Random => {
            println!("Random dataset is generated automatically, no download needed.");
        }
        DatasetType::Sift1M => {
            let info = DatasetInfo::sift1m();
            println!("To download the {} dataset:", info.name);
            println!();
            println!("  # Option 1: Direct download");
            println!("  wget {}sift.tar.gz", info.base_url);
            println!("  tar -xzf sift.tar.gz");
            println!();
            println!("  # Option 2: From texmex mirror");
            println!("  curl -O http://corpus-texmex.irisa.fr/sift.tar.gz");
            println!("  tar -xzf sift.tar.gz");
            println!();
            println!("Expected files after extraction:");
            println!("  sift/sift_base.fvecs      (1M vectors, 128-dim, ~512MB)");
            println!("  sift/sift_query.fvecs     (10K vectors)");
            println!("  sift/sift_groundtruth.ivecs (10K x 100 nearest neighbors)");
        }
        DatasetType::Sift100M => {
            println!("To download the SIFT100M (BigANN/ANN_SIFT1B) dataset:");
            println!();
            println!("  # Download base vectors (~100GB compressed, ~128GB uncompressed)");
            println!("  wget ftp://ftp.irisa.fr/local/texmex/corpus/bigann_base.bvecs.gz");
            println!("  gunzip bigann_base.bvecs.gz");
            println!();
            println!("  # Download query vectors");
            println!("  wget ftp://ftp.irisa.fr/local/texmex/corpus/bigann_query.bvecs.gz");
            println!("  gunzip bigann_query.bvecs.gz");
            println!();
            println!("  # Download ground truth");
            println!("  wget ftp://ftp.irisa.fr/local/texmex/corpus/bigann_gnd.tar.gz");
            println!("  tar -xzf bigann_gnd.tar.gz");
            println!();
            println!("Expected files after extraction:");
            println!("  bigann_base.bvecs     (1B vectors, 128-dim, uint8 format, ~128GB)");
            println!("  bigann_query.bvecs    (10K query vectors)");
            println!("  gnd/idx_1M.ivecs      (ground truth for 1M subset)");
            println!("  gnd/idx_10M.ivecs     (ground truth for 10M subset)");
            println!("  gnd/idx_100M.ivecs    (ground truth for 100M subset)");
            println!("  gnd/idx_1000M.ivecs   (ground truth for 1B)");
            println!();
            println!("Note: Use -n to limit vectors (e.g., -n 10000000 for 10M).");
            println!("Ground truth file is auto-selected based on -n value.");
        }
        DatasetType::Deep1M => {
            let info = DatasetInfo::deep1m();
            println!("To download the {} dataset:", info.name);
            println!();
            println!("  # From ann-benchmarks or Yandex");
            println!("  # DEEP1B is very large; you typically want a 1M subset");
            println!();
            println!("  # Option 1: From ann-benchmarks (if available)");
            println!("  python -m ann_benchmarks.datasets --dataset deep-image-96-angular");
            println!();
            println!("  # Option 2: Generate from DEEP1B");
            println!("  wget {}base.1B.fbin", info.base_url);
            println!("  # Then extract first 1M vectors");
            println!();
            println!("Expected files:");
            println!("  deep1m_base.fvecs or base.1M.fbin  (1M vectors, 96-dim)");
            println!("  deep1m_query.fvecs                 (10K vectors)");
            println!("  deep1m_groundtruth.ivecs           (10K x 100 nearest neighbors)");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::tempdir;

    #[test]
    fn test_read_write_fvecs() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.fvecs");

        // Write test vectors
        let vectors: Vec<Vec<f32>> = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
        ];

        {
            let mut file = File::create(&path).unwrap();
            for v in &vectors {
                let dim = v.len() as i32;
                file.write_all(&dim.to_le_bytes()).unwrap();
                for &val in v {
                    file.write_all(&val.to_le_bytes()).unwrap();
                }
            }
        }

        // Read back
        let loaded = read_fvecs(&path).unwrap();
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0], vec![1.0, 2.0, 3.0]);
        assert_eq!(loaded[1], vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_read_ivecs() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.ivecs");

        // Write test vectors
        let vectors: Vec<Vec<u32>> = vec![
            vec![10, 20, 30],
            vec![40, 50, 60],
        ];

        {
            let mut file = File::create(&path).unwrap();
            for v in &vectors {
                let dim = v.len() as i32;
                file.write_all(&dim.to_le_bytes()).unwrap();
                for &val in v {
                    file.write_all(&val.to_le_bytes()).unwrap();
                }
            }
        }

        // Read back
        let loaded = read_ivecs(&path).unwrap();
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0], vec![10, 20, 30]);
        assert_eq!(loaded[1], vec![40, 50, 60]);
    }

    #[test]
    fn test_dataset_type_parsing() {
        assert_eq!("random".parse::<DatasetType>().unwrap(), DatasetType::Random);
        assert_eq!("sift1m".parse::<DatasetType>().unwrap(), DatasetType::Sift1M);
        assert_eq!("SIFT".parse::<DatasetType>().unwrap(), DatasetType::Sift1M);
        assert_eq!("deep1m".parse::<DatasetType>().unwrap(), DatasetType::Deep1M);
        assert_eq!("DEEP".parse::<DatasetType>().unwrap(), DatasetType::Deep1M);
    }
}
