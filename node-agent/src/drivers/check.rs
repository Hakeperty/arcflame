use std::process::Command;
use crate::network::arcflare as pb;
use pb::{DriverInfo, DriverReport};

pub async fn audit() -> DriverReport {
    let mut drivers = Vec::new();
    let mut recommendations = Vec::new();

    // Check NVIDIA
    if let Some(info) = check_nvidia().await {
        if info.status == "proprietary" {
            recommendations.push("NVIDIA proprietary driver detected — good".to_string());
        } else if info.status == "nouveau" {
            recommendations.push(
                "Nouveau driver detected. Install NVIDIA proprietary driver for 2-5x faster inference: sudo apt install nvidia-driver-550".to_string()
            );
        } else {
            recommendations.push(
                "No NVIDIA driver found. If you have an NVIDIA GPU: sudo apt install nvidia-driver-550".to_string()
            );
        }
        drivers.push(info);
    }

    // Check AMD
    if let Some(info) = check_amd().await {
        if info.current_driver == "amdgpu" {
            recommendations.push("AMDGPU driver active — good".to_string());
        } else if info.current_driver == "radeon" {
            recommendations.push(
                "Legacy Radeon driver detected. Switch to AMDGPU for better performance: add 'amdgpu.si_support=1 amdgpu.cik_support=1' to kernel cmdline".to_string()
            );
        }
        drivers.push(info);
    }

    // Check Intel
    if let Some(info) = check_intel().await {
        drivers.push(info);
    }

    let missing = check_missing_firmware().await;

    DriverReport {
        drivers,
        missing_firmware: missing,
        recommendations,
        up_to_date: true,
    }
}

async fn check_nvidia() -> Option<DriverInfo> {
    // Check if nvidia-smi works (proprietary driver)
    if let Ok(output) = Command::new("nvidia-smi")
        .args(["--query-gpu=driver_version", "--format=csv,noheader"])
        .output()
    {
        if output.status.success() {
            let version = String::from_utf8_lossy(&output.stdout)
                .lines().next().unwrap_or("unknown")
                .trim().to_string();

            return Some(DriverInfo {
                device: "NVIDIA GPU".to_string(),
                current_driver: format!("nvidia ({})", version),
                latest_available: "550+".to_string(),
                status: "proprietary".to_string(),
                recommendation: "Up-to-date".to_string(),
            });
        }
    }

    // Check if nouveau is loaded
    if let Ok(output) = Command::new("lsmod").output() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        if stdout.contains("nouveau") {
            return Some(DriverInfo {
                device: "NVIDIA GPU".to_string(),
                current_driver: "nouveau (open source)".to_string(),
                latest_available: "nvidia-550".to_string(),
                status: "nouveau".to_string(),
                recommendation: "Install proprietary driver for 2-5x faster inference".to_string(),
            });
        }
    }

    // No NVIDIA hardware detected
    None
}

async fn check_amd() -> Option<DriverInfo> {
    if !std::path::Path::new("/sys/module/amdgpu").exists() &&
       !std::path::Path::new("/sys/module/radeon").exists() {
        return None;
    }

    if std::path::Path::new("/sys/module/amdgpu").exists() {
        let version = std::fs::read_to_string("/sys/module/amdgpu/version")
            .unwrap_or_else(|_| "kernel-builtin".to_string());

        Some(DriverInfo {
            device: "AMD GPU".to_string(),
            current_driver: format!("amdgpu ({})", version.trim()),
            latest_available: "amdgpu (latest)".to_string(),
            status: "open".to_string(),
            recommendation: "Good — AMDGPU active".to_string(),
        })
    } else {
        Some(DriverInfo {
            device: "AMD GPU".to_string(),
            current_driver: "radeon (legacy)".to_string(),
            latest_available: "amdgpu".to_string(),
            status: "legacy".to_string(),
            recommendation: "Switch to AMDGPU for 20-30% better performance".to_string(),
        })
    }
}

async fn check_intel() -> Option<DriverInfo> {
    if std::path::Path::new("/sys/module/i915").exists() {
        Some(DriverInfo {
            device: "Intel GPU".to_string(),
            current_driver: "i915".to_string(),
            latest_available: "i915".to_string(),
            status: "open".to_string(),
            recommendation: "Intel iGPU — limited performance for LLM".to_string(),
        })
    } else {
        None
    }
}

async fn check_missing_firmware() -> Vec<String> {
    let mut missing = Vec::new();

    if let Ok(output) = Command::new("dmesg").output() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        for line in stdout.lines() {
            if line.contains("firmware") && (line.contains("failed") || line.contains("missing")) {
                // Extract the firmware name
                let parts: Vec<&str> = line.split_whitespace().collect();
                for part in parts {
                    if part.contains(".bin") {
                        missing.push(part.to_string());
                    }
                }
            }
        }
    }

    missing
}
