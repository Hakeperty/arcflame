use crate::network::arcflare as pb;
use pb::MemoryInfo;
use sysinfo::System;

pub async fn detect() -> MemoryInfo {
    let mut sys = System::new_all();
    sys.refresh_memory();

    let total_bytes = sys.total_memory();
    let available_bytes = sys.available_memory();
    let swap_total = sys.total_swap();
    let swap_free = sys.free_swap();

    // Hugepages info
    let (hugepage_size_kb, hugepages_total, hugepages_free) = hugepage_info();

    MemoryInfo {
        total_bytes,
        available_bytes,
        swap_total_bytes: swap_total,
        swap_available_bytes: swap_free,
        hugepage_size_kb,
        hugepages_total,
        hugepages_free,
    }
}

fn hugepage_info() -> (u64, u64, u64) {
    let size = std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|c| {
            c.lines().find_map(|l| {
                l.strip_prefix("Hugepagesize:").and_then(|s| {
                    s.trim().trim_end_matches(" kB").parse::<u64>().ok()
                })
            })
        })
        .unwrap_or(0);

    let total = std::fs::read_to_string("/proc/sys/vm/nr_hugepages")
        .ok()
        .and_then(|c| c.trim().parse::<u64>().ok())
        .unwrap_or(0);

    let free = std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|c| {
            c.lines().find_map(|l| {
                l.strip_prefix("HugePages_Free:").and_then(|s| {
                    s.trim().parse::<u64>().ok()
                })
            })
        })
        .unwrap_or(0);

    (size, total, free)
}
