use crate::network::arcflare as pb;
use pb::BenchmarkResult;
use std::time::Instant;

pub async fn run_cpu_benchmark() -> BenchmarkResult {
    let start = Instant::now();
    let iterations = 100_000_000u64;

    for i in 0..iterations {
        let _ = (i as f64).sin() * (i as f64).cos();
    }

    let elapsed = start.elapsed();
    let ops_per_sec = iterations as f64 / elapsed.as_secs_f64();

    BenchmarkResult {
        score: ops_per_sec / 1_000_000.0,
        tokens_per_second: 0.0,
        memory_bandwidth_gbps: 0.0,
        compute_flops: ops_per_sec,
        details: std::collections::HashMap::new(),
    }
}

#[allow(dead_code)]
pub async fn run_memory_benchmark() -> BenchmarkResult {
    let size = 256 * 1024 * 1024; // 256 MB
    let mut buf = vec![0u8; size];

    let start = Instant::now();
    for i in 0..buf.len() {
        buf[i] = (i % 256) as u8;
    }
    let elapsed = start.elapsed();

    let bytes_per_sec = size as f64 / elapsed.as_secs_f64();
    let bandwidth = bytes_per_sec / (1024.0 * 1024.0 * 1024.0);

    BenchmarkResult {
        score: bandwidth,
        tokens_per_second: 0.0,
        memory_bandwidth_gbps: bandwidth,
        compute_flops: 0.0,
        details: std::collections::HashMap::new(),
    }
}
