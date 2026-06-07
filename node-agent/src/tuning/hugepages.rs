#![allow(dead_code)]
pub async fn pre_allocate(count: u64) -> Result<(), String> {
    std::fs::write("/proc/sys/vm/nr_hugepages", count.to_string())
        .map_err(|e| format!("Failed to allocate hugepages (need root?): {}", e))
}

pub async fn get_hugepage_size() -> u64 {
    std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|c| {
            c.lines().find_map(|l| {
                l.strip_prefix("Hugepagesize:").and_then(|s| {
                    s.trim().trim_end_matches(" kB").parse::<u64>().ok()
                })
            })
        })
        .unwrap_or(0)
}

pub async fn get_current_count() -> u64 {
    std::fs::read_to_string("/proc/sys/vm/nr_hugepages")
        .ok()
        .and_then(|c| c.trim().parse::<u64>().ok())
        .unwrap_or(0)
}
