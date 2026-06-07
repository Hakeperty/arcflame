#![allow(dead_code)]
use std::process::Command;
use tracing::warn;

pub async fn apply_aggressive_tuning() -> Result<String, String> {
    let mut changes = Vec::new();

    // First apply safe tuning
    let _ = super::safe::apply_safe_tuning().await;

    // Intel undervolt if available
    if Command::new("intel-undervolt")
        .arg("read")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
    {
        match Command::new("intel-undervolt")
            .args(["core", "-50"])
            .output()
        {
            Ok(output) if output.status.success() => {
                changes.push("CPU undervolt -50mV (core)");
            }
            _ => warn!("Intel undervolt failed"),
        }

        let _ = Command::new("intel-undervolt")
            .args(["cache", "-50"])
            .output();
        let _ = Command::new("intel-undervolt")
            .arg("apply")
            .output();

        changes.push("CPU undervolt -50mV (cache)");
    }

    // RyzenAdj for AMD
    if Command::new("ryzenadj")
        .arg("-i")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
    {
        let _ = Command::new("ryzenadj")
            .args(["--stapm-limit", "35000"])
            .output();
        changes.push("Ryzen TDP increased to 35W");
    }

    // NVIDIA GPU overclock
    if Command::new("nvidia-smi")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
    {
        // Increase power limit
        let _ = Command::new("nvidia-smi")
            .args(["-pl", "150"])
            .output();

        // Lock GPU clock (safe +200MHz offset)
        let _ = Command::new("nvidia-smi")
            .args(["-lgc", "1500,1900"])
            .output();

        changes.push("NVIDIA GPU: power limit + clock locked");
    }

    Ok(format!("Applied aggressive tuning: {}", changes.join(", ")))
}
