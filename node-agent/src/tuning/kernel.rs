#![allow(dead_code)]
use std::process::Command;

pub async fn set_performance_governor() -> Result<(), String> {
    Command::new("cpupower")
        .args(["frequency-set", "-g", "performance"])
        .output()
        .map_err(|e| format!("cpupower not found: {}", e))
        .and_then(|o| {
            if o.status.success() {
                Ok(())
            } else {
                Err("Failed to set governor (need root?)".to_string())
            }
        })
}

pub async fn set_swappiness(value: u32) -> Result<(), String> {
    let val = value.to_string();
    std::fs::write("/proc/sys/vm/swappiness", &val)
        .map_err(|e| format!("Failed to set swappiness: {}", e))
}

pub async fn set_thp_mode(mode: &str) -> Result<(), String> {
    std::fs::write("/sys/kernel/mm/transparent_hugepage/enabled", mode)
        .map_err(|e| format!("Failed to set THP: {}", e))
}

pub async fn disable_numa_balancing() -> Result<(), String> {
    std::fs::write("/proc/sys/kernel/numa_balancing", "0")
        .map_err(|e| format!("Failed to disable NUMA balancing: {}", e))
}

pub async fn tune_all() -> Result<Vec<String>, String> {
    let mut applied = Vec::new();

    if set_performance_governor().await.is_ok() {
        applied.push("CPU governor → performance".to_string());
    }
    if set_swappiness(10).await.is_ok() {
        applied.push("Swappiness → 10".to_string());
    }
    if disable_numa_balancing().await.is_ok() {
        applied.push("NUMA balancing disabled".to_string());
    }

    Ok(applied)
}
