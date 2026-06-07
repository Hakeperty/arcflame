use tracing::{info, warn};

pub async fn set_scheduler(scheduler: &str) -> Result<(), String> {
    // Find NVMe or SSD devices
    let block_dir = "/sys/block";
    let dir = std::path::Path::new(block_dir);

    if !dir.exists() {
        return Err("No block devices found".to_string());
    }

    let mut applied = false;

    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name_str = name.to_string_lossy();

            // Only target non-rotational (SSD/NVMe) devices
            let rotational_path = entry.path().join("queue/rotational");
            if let Ok(rotational) = std::fs::read_to_string(&rotational_path) {
                if rotational.trim() == "0" {
                    let sched_path = entry.path().join("queue/scheduler");
                    if let Ok(content) = std::fs::read_to_string(&sched_path) {
                        if content.contains(scheduler) {
                            if std::fs::write(&sched_path, scheduler).is_ok() {
                                info!("Set I/O scheduler on {} to {}", name_str, scheduler);
                                applied = true;
                            }
                        }
                    }
                }
            }
        }
    }

    if applied {
        Ok(())
    } else {
        warn!("Could not set I/O scheduler to {} on any device", scheduler);
        Err(format!("Failed to set I/O scheduler to {}", scheduler))
    }
}
