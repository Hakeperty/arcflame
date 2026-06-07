pub mod cpu;
pub mod gpu;
pub mod memory;
pub mod benchmark;

use super::network::arcflare as pb;
use pb::{DiskInfo, HardwareReport, NetworkInfo};

pub async fn collect() -> anyhow::Result<HardwareReport> {
    let cpu = cpu::detect().await;
    let gpus = gpu::detect().await;
    let memory = memory::detect().await;
    let disk = disk_info().await;
    let network = network_info().await;

    Ok(HardwareReport {
        node_id: String::new(),
        cpu: Some(cpu),
        gpus,
        memory: Some(memory),
        disk: Some(disk),
        network: Some(network),
        benchmark_score: 0.0,
    })
}

async fn disk_info() -> DiskInfo {
    use sysinfo::Disks;

    let disks = Disks::new();
    let total: u64 = disks.iter().map(|d| d.total_space()).sum();
    let available: u64 = disks.iter().map(|d| d.available_space()).sum();

    let is_ssd = disks.iter().any(|d| {
        d.kind() == sysinfo::DiskKind::SSD
    });

    DiskInfo {
        total_bytes: total,
        available_bytes: available,
        is_ssd,
        read_speed_mbps: 0.0,
        write_speed_mbps: 0.0,
    }
}

async fn network_info() -> NetworkInfo {
    let hostname = std::fs::read_to_string("/etc/hostname")
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|_| "unknown".to_string());

    let ips = local_ip_addresses();

    NetworkInfo {
        hostname,
        ip_addresses: ips,
        discovery_port: 0,
        links: vec![],
    }
}

fn local_ip_addresses() -> Vec<String> {
    let mut ips = Vec::new();
    // Try common interfaces via reading /proc/net/fib_trie or just use hostname -I
    if let Ok(output) = std::process::Command::new("hostname")
        .arg("-I")
        .output()
    {
        let stdout = String::from_utf8_lossy(&output.stdout);
        for ip in stdout.split_whitespace() {
            if !ip.is_empty() {
                ips.push(ip.to_string());
            }
        }
    }
    if !ips.is_empty() {
        return ips;
    }
    // Fallback: try reading /proc/net/fib_trie
    if let Ok(content) = std::fs::read_to_string("/proc/net/fib_trie") {
        for line in content.lines() {
            if let Some(ip) = line.trim().strip_prefix("+-- ") {
                if ip.contains('.') && !ip.starts_with("127.") && !ip.starts_with("0.") {
                    let ip = ip.split('/').next().unwrap_or(ip);
                    ips.push(ip.to_string());
                }
            }
        }
    }
    ips
}
