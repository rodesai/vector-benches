use clap::Parser;
use std::time::Instant;
use usearch_bench::{
    build_index_parallel, generate_partitioned_vectors, load_from_buffer, serialize_to_buffer,
    BenchConfig, DEFAULT_CONNECTIVITY, DEFAULT_DIMENSIONS, DEFAULT_EXPANSION_ADD,
    DEFAULT_EXPANSION_SEARCH, DEFAULT_NUM_THREADS, DEFAULT_NUM_VECTORS,
};

#[derive(Parser, Debug)]
#[command(name = "usearch-bench")]
#[command(about = "Benchmark usearch index operations")]
struct Args {
    /// Number of vectors to index
    #[arg(short = 'n', long, default_value_t = DEFAULT_NUM_VECTORS)]
    num_vectors: usize,

    /// Vector dimensions
    #[arg(short = 'd', long, default_value_t = DEFAULT_DIMENSIONS)]
    dimensions: usize,

    /// Graph connectivity (edges per node)
    #[arg(short = 'c', long, default_value_t = DEFAULT_CONNECTIVITY)]
    connectivity: usize,

    /// Expansion factor during index construction
    #[arg(long, default_value_t = DEFAULT_EXPANSION_ADD)]
    expansion_add: usize,

    /// Expansion factor during search
    #[arg(long, default_value_t = DEFAULT_EXPANSION_SEARCH)]
    expansion_search: usize,

    /// Random seed for reproducibility
    #[arg(short = 's', long, default_value_t = 42)]
    seed: u64,

    /// Number of threads for parallel index building (0 = use all available cores)
    #[arg(short = 't', long, default_value_t = DEFAULT_NUM_THREADS)]
    threads: usize,
}

fn format_duration(duration: std::time::Duration) -> String {
    let secs = duration.as_secs_f64();
    if secs >= 1.0 {
        format!("{:.3} s", secs)
    } else if secs >= 0.001 {
        format!("{:.3} ms", secs * 1000.0)
    } else {
        format!("{:.3} µs", secs * 1_000_000.0)
    }
}

fn format_bytes(bytes: usize) -> String {
    if bytes >= 1024 * 1024 * 1024 {
        format!("{:.2} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    } else if bytes >= 1024 * 1024 {
        format!("{:.2} MB", bytes as f64 / (1024.0 * 1024.0))
    } else if bytes >= 1024 {
        format!("{:.2} KB", bytes as f64 / 1024.0)
    } else {
        format!("{} bytes", bytes)
    }
}

fn format_throughput(count: usize, duration: std::time::Duration) -> String {
    let per_second = count as f64 / duration.as_secs_f64();
    if per_second >= 1_000_000.0 {
        format!("{:.2} M/s", per_second / 1_000_000.0)
    } else if per_second >= 1_000.0 {
        format!("{:.2} K/s", per_second / 1_000.0)
    } else {
        format!("{:.2}/s", per_second)
    }
}

fn main() {
    let args = Args::parse();

    let config = BenchConfig {
        num_vectors: args.num_vectors,
        dimensions: args.dimensions,
        connectivity: args.connectivity,
        expansion_add: args.expansion_add,
        expansion_search: args.expansion_search,
        num_threads: args.threads,
        ..Default::default()
    };

    let effective_threads = config.effective_threads();

    println!("=== USearch Benchmark ===");
    println!();
    println!("Configuration:");
    println!("  Vectors:        {:>12}", args.num_vectors);
    println!("  Dimensions:     {:>12}", args.dimensions);
    println!("  Connectivity:   {:>12}", args.connectivity);
    println!("  Expansion (add):{:>12}", args.expansion_add);
    println!("  Expansion (search):{:>9}", args.expansion_search);
    println!("  Threads:        {:>12}", effective_threads);
    println!("  Seed:           {:>12}", args.seed);
    println!();

    // Generate random vectors (partitioned for multi-threaded building)
    print!(
        "Generating {} random vectors ({} partitions)... ",
        args.num_vectors, effective_threads
    );
    std::io::Write::flush(&mut std::io::stdout()).unwrap();
    let start = Instant::now();
    let partitioned_vectors = generate_partitioned_vectors(
        config.num_vectors,
        config.dimensions,
        effective_threads,
        args.seed,
    );
    let gen_duration = start.elapsed();
    println!("done ({})", format_duration(gen_duration));
    println!();

    // Benchmark: Build index (parallel)
    println!("--- Build Index ({} threads) ---", effective_threads);
    let start = Instant::now();
    let index = build_index_parallel(&config, &partitioned_vectors);
    let build_duration = start.elapsed();
    println!(
        "  Time:           {:>12}",
        format_duration(build_duration)
    );
    println!(
        "  Throughput:     {:>12}",
        format_throughput(args.num_vectors, build_duration)
    );
    println!(
        "  Memory usage:   {:>12}",
        format_bytes(index.memory_usage())
    );
    println!();

    // Benchmark: Serialize to buffer
    println!("--- Serialize to Buffer ---");
    let serialized_size = index.serialized_length();
    println!(
        "  Buffer size:    {:>12}",
        format_bytes(serialized_size)
    );
    let start = Instant::now();
    let buffer = serialize_to_buffer(&index);
    let serialize_duration = start.elapsed();
    println!(
        "  Time:           {:>12}",
        format_duration(serialize_duration)
    );
    let serialize_throughput = buffer.len() as f64 / serialize_duration.as_secs_f64();
    println!(
        "  Throughput:     {:>12}/s",
        format_bytes(serialize_throughput as usize)
    );
    println!();

    // Benchmark: Load from buffer
    println!("--- Load from Buffer ---");
    let start = Instant::now();
    let loaded_index = load_from_buffer(&config, &buffer);
    let load_duration = start.elapsed();
    println!("  Time:           {:>12}", format_duration(load_duration));
    let load_throughput = buffer.len() as f64 / load_duration.as_secs_f64();
    println!(
        "  Throughput:     {:>12}/s",
        format_bytes(load_throughput as usize)
    );
    println!("  Loaded vectors: {:>12}", loaded_index.size());
    println!();

    // Summary
    println!("=== Summary ===");
    println!(
        "  Total vectors:  {:>12}",
        args.num_vectors
    );
    println!(
        "  Build time:     {:>12}",
        format_duration(build_duration)
    );
    println!(
        "  Serialize time: {:>12}",
        format_duration(serialize_duration)
    );
    println!(
        "  Load time:      {:>12}",
        format_duration(load_duration)
    );
}
