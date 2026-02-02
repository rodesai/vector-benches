use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use usearch_bench::{
    build_index, generate_random_vectors, load_from_buffer, serialize_to_buffer, BenchConfig,
};

/// Get the number of vectors from environment variable, or use default
fn get_num_vectors() -> usize {
    std::env::var("USEARCH_NUM_VECTORS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(usearch_bench::DEFAULT_NUM_VECTORS)
}

fn bench_build_index(c: &mut Criterion) {
    let num_vectors = get_num_vectors();
    let config = BenchConfig::default().with_num_vectors(num_vectors);

    // Pre-generate vectors outside the benchmark
    let vectors = generate_random_vectors(config.num_vectors, config.dimensions, 42);

    let mut group = c.benchmark_group("index_build");
    group.throughput(Throughput::Elements(num_vectors as u64));
    group.sample_size(10); // Reduce sample size for large indexes

    group.bench_with_input(
        BenchmarkId::new("build", format!("{} vectors", num_vectors)),
        &vectors,
        |b, vectors| {
            b.iter(|| build_index(&config, vectors));
        },
    );

    group.finish();
}

fn bench_serialize_index(c: &mut Criterion) {
    let num_vectors = get_num_vectors();
    let config = BenchConfig::default().with_num_vectors(num_vectors);

    // Pre-build index outside the benchmark
    let vectors = generate_random_vectors(config.num_vectors, config.dimensions, 42);
    let index = build_index(&config, &vectors);

    let serialized_size = index.serialized_length();

    let mut group = c.benchmark_group("index_serialize");
    group.throughput(Throughput::Bytes(serialized_size as u64));
    group.sample_size(10);

    group.bench_with_input(
        BenchmarkId::new("serialize_to_buffer", format!("{} vectors", num_vectors)),
        &index,
        |b, index| {
            b.iter(|| serialize_to_buffer(index));
        },
    );

    group.finish();
}

fn bench_load_index(c: &mut Criterion) {
    let num_vectors = get_num_vectors();
    let config = BenchConfig::default().with_num_vectors(num_vectors);

    // Pre-build and serialize index outside the benchmark
    let vectors = generate_random_vectors(config.num_vectors, config.dimensions, 42);
    let index = build_index(&config, &vectors);
    let buffer = serialize_to_buffer(&index);

    let mut group = c.benchmark_group("index_load");
    group.throughput(Throughput::Bytes(buffer.len() as u64));
    group.sample_size(10);

    group.bench_with_input(
        BenchmarkId::new("load_from_buffer", format!("{} vectors", num_vectors)),
        &buffer,
        |b, buffer| {
            b.iter(|| load_from_buffer(&config, buffer));
        },
    );

    group.finish();
}

criterion_group!(
    benches,
    bench_build_index,
    bench_serialize_index,
    bench_load_index
);
criterion_main!(benches);
