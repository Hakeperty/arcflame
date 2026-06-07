use std::process::Command;
use tracing::{info, warn};

pub async fn apply_safe_tuning() -> Result<String, String> {
    let mut changes = Vec::new();

    // Set performance governor on all CPUs
    match Command::new("cpupower")
        .args(["frequency-set", "-g", "performance"])
        .output()
    {
        Ok(output) if output.status.success() => {
            changes.push("CPU governor → performance");
            info!("CPU governor set to performance");
        }
        _ => warn!("Failed to set CPU governor (may need root)"),
    }

    // Disable turbo boost (helps stability on old hardware)
    if let Ok(content) = std::fs::read_to_string("/sys/devices/system/cpu/intel_pstate/no_turbo") {
        if content.trim() == "0" {
            let _ = std::fs::write(
                "/sys/devices/system/cpu/intel_pstate/no_turbo",
                "1",
            );
            changes.push("Turbo boost disabled");
        }
    }

    // Set HWP max perf to 100% (Intel)
    let _ = std::fs::write(
        "/sys/devices/system/cpu/intel_pstate/max_perf_pct",
        "100",
    );

    Ok(format!("Applied safe tuning: {}", changes.join(", ")))
}
