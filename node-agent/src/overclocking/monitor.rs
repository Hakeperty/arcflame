use std::process::Command;
use crate::network::arcflare as pb;
use pb::OverclockingStatus;

pub async fn get_status() -> OverclockingStatus {
    let temperature = measure_temperature().await;
    let throttling = check_throttling().await;
    let cpu_freq = read_cpu_freq().await;
    let gpu_freq = read_gpu_freq().await;

    OverclockingStatus {
        current_mode: String::new(),
        temperature_celsius: temperature,
        is_throttling: throttling,
        cpu_freq_ghz: cpu_freq,
        gpu_freq_mhz: gpu_freq,
        uptime_seconds: 0,
    }
}

pub async fn measure_temperature() -> f64 {
    // Try CPU temperature via sysfs
    let temp_paths = [
        "/sys/class/thermal/thermal_zone0/temp",
        "/sys/class/hwmon/hwmon0/temp1_input",
        "/sys/class/hwmon/hwmon1/temp1_input",
    ];

    for path in &temp_paths {
        if let Ok(content) = std::fs::read_to_string(path) {
            if let Ok(millicelsius) = content.trim().parse::<f64>() {
                return millicelsius / 1000.0;
            }
        }
    }

    // Fallback: try `sensors` command
    if let Ok(output) = Command::new("sensors")
        .arg("-u")
        .output()
    {
        let stdout = String::from_utf8_lossy(&output.stdout);
        for line in stdout.lines() {
            if line.contains("_input:") {
                if let Some(val) = line.split(':').nth(1) {
                    if let Ok(temp) = val.trim().parse::<f64>() {
                        return temp;
                    }
                }
            }
        }
    }

    0.0
}

pub async fn check_throttling() -> bool {
    // Check for thermal throttling indicators
    if let Ok(content) = std::fs::read_to_string("/sys/devices/system/cpu/cpu0/thermal_throttle/core_throttle_count") {
        if let Ok(count) = content.trim().parse::<u64>() {
            return count > 0;
        }
    }
    false
}

pub async fn read_cpu_freq() -> f64 {
    if let Ok(freq) = std::fs::read_to_string(
        "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq"
    ) {
        if let Ok(khz) = freq.trim().parse::<f64>() {
            return khz / 1_000_000.0;
        }
    }
    0.0
}

pub async fn read_gpu_freq() -> f64 {
    // Try NVIDIA
    if let Ok(output) = Command::new("nvidia-smi")
        .args(["--query-gpu=clocks.current.graphics", "--format=csv,noheader"])
        .output()
    {
        let stdout = String::from_utf8_lossy(&output.stdout);
        if let Some(line) = stdout.lines().next() {
            let clean = line.trim().trim_end_matches(" MHz");
            if let Ok(mhz) = clean.parse::<f64>() {
                return mhz;
            }
        }
    }
    0.0
}
