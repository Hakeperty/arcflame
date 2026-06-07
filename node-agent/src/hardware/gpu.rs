use crate::network::arcflare as pb;
use pb::GpuInfo;

pub async fn detect() -> Vec<GpuInfo> {
    let mut gpus = Vec::new();

    // NVIDIA via nvidia-smi if available
    if let Some(gpu) = detect_nvidia().await {
        gpus.push(gpu);
    }

    // AMD via sysfs / lspci
    if let Some(gpu) = detect_amd().await {
        gpus.push(gpu);
    }

    // Intel integrated
    if let Some(gpu) = detect_intel().await {
        gpus.push(gpu);
    }

    gpus
}

async fn detect_nvidia() -> Option<GpuInfo> {
    let output = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    for line in stdout.lines() {
        let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
        if parts.len() >= 2 {
            let vram = parse_vram(parts.get(1).unwrap_or(&"0"));
            return Some(GpuInfo {
                name: parts[0].to_string(),
                vendor: "NVIDIA".to_string(),
                driver: parts.get(2).unwrap_or(&"unknown").to_string(),
                vram_bytes: vram,
                available: true,
                compute_benchmark: 0.0,
            });
        }
    }

    None
}

async fn detect_amd() -> Option<GpuInfo> {
    // Check if amdgpu driver is loaded
    if !path_exists("/sys/module/amdgpu") {
        return None;
    }

    // Try to get name from lspci
    let output = std::process::Command::new("lspci")
        .args(["-nn"])
        .output()
        .ok()?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    for line in stdout.lines() {
        if line.contains("VGA") && (line.contains("AMD") || line.contains("Advanced Micro")) {
            let name = line.split(": ").nth(1).unwrap_or("AMD GPU").to_string();
            let vram = detect_amd_vram().await;
            return Some(GpuInfo {
                name,
                vendor: "AMD".to_string(),
                driver: "amdgpu".to_string(),
                vram_bytes: vram,
                available: true,
                compute_benchmark: 0.0,
            });
        }
    }

    None
}

async fn detect_amd_vram() -> u64 {
    // Try sysfs for AMD VRAM size
    let paths = [
        "/sys/class/drm/card0/device/mem_info_vram_total",
        "/sys/class/drm/card1/device/mem_info_vram_total",
    ];

    for path in &paths {
        if let Ok(content) = std::fs::read_to_string(path) {
            if let Ok(bytes) = content.trim().parse::<u64>() {
                return bytes;
            }
        }
    }
    0
}

async fn detect_intel() -> Option<GpuInfo> {
    if !path_exists("/sys/module/i915") {
        return None;
    }

    let name = "Intel Integrated Graphics".to_string();
    let vram = detect_intel_vram().await;

    Some(GpuInfo {
        name,
        vendor: "Intel".to_string(),
        driver: "i915".to_string(),
        vram_bytes: vram,
        available: true,
        compute_benchmark: 0.0,
    })
}

async fn detect_intel_vram() -> u64 {
    if let Ok(freq) = std::fs::read_to_string("/sys/class/drm/card0/gt_cur_freq_mhz") {
        if let Ok(_mhz) = freq.trim().parse::<f64>() {
            // Intel iGPUs use shared memory; report a portion of system RAM
            let sys = sysinfo::System::new();
            let total = sys.total_memory();
            // iGPUs typically reserve 128MB-2GB of system RAM
            return std::cmp::min(total / 8, 2 * 1024 * 1024 * 1024);
        }
    }
    0
}

fn path_exists(path: &str) -> bool {
    std::path::Path::new(path).exists()
}

fn parse_vram(s: &str) -> u64 {
    let s = s.trim().to_lowercase();
    if s.ends_with("mib") {
        let n: f64 = s.trim_end_matches("mib").trim().parse().unwrap_or(0.0);
        (n * 1024.0 * 1024.0) as u64
    } else if s.ends_with("gib") {
        let n: f64 = s.trim_end_matches("gib").trim().parse().unwrap_or(0.0);
        (n * 1024.0 * 1024.0 * 1024.0) as u64
    } else {
        0
    }
}
