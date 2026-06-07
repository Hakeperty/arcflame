use crate::network::arcflare as pb;
use pb::CpuInfo;

pub async fn detect() -> CpuInfo {
    let mut cpu_info = CpuInfo {
        model: String::new(),
        cores: 0,
        threads: 0,
        max_freq_ghz: 0.0,
        min_freq_ghz: 0.0,
        architecture: std::env::consts::ARCH.to_string(),
        has_avx2: false,
        has_avx512: false,
        has_neon: false,
        governor: String::new(),
    };

    // Read /proc/cpuinfo for model and features
    if let Ok(contents) = std::fs::read_to_string("/proc/cpuinfo") {
        for line in contents.lines() {
            if let Some(model) = line.strip_prefix("model name\t: ") {
                cpu_info.model = model.to_string();
            }
            if let Some(features) = line.strip_prefix("flags\t\t: ") {
                cpu_info.has_avx2 = features.contains("avx2");
                cpu_info.has_avx512 = features.contains("avx512f");
            }
            if let Some(features) = line.strip_prefix("Features\t: ") {
                cpu_info.has_neon = features.contains("neon");
            }
        }
    }

    // CPU count via sysinfo
    let sys = sysinfo::System::new();
    cpu_info.cores = sys.physical_core_count().unwrap_or(0) as u32;
    cpu_info.threads = sys.cpus().len() as u32;

    // Frequency from sysfs
    if let Ok(freq_str) = std::fs::read_to_string(
        "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq"
    ) {
        if let Ok(khz) = freq_str.trim().parse::<f64>() {
            cpu_info.max_freq_ghz = khz / 1_000_000.0;
        }
    }

    if let Ok(freq_str) = std::fs::read_to_string(
        "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_min_freq"
    ) {
        if let Ok(khz) = freq_str.trim().parse::<f64>() {
            cpu_info.min_freq_ghz = khz / 1_000_000.0;
        }
    }

    // Governor
    if let Ok(gov) = std::fs::read_to_string(
        "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"
    ) {
        cpu_info.governor = gov.trim().to_string();
    }

    cpu_info
}
