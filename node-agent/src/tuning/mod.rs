pub mod kernel;
pub mod hugepages;
pub mod io;

use crate::network::arcflare as pb;
use pb::{TuningRequest, TuningResponse};

pub async fn apply_tuning(req: TuningRequest) -> TuningResponse {
    let mut changes = Vec::new();
    let mut errors = Vec::new();

    if req.set_performance_governor {
        match kernel::set_performance_governor().await {
            Ok(_) => changes.push("CPU governor → performance".to_string()),
            Err(e) => errors.push(e),
        }
    }

    if req.set_hugepages {
        match hugepages::pre_allocate(req.hugepages_count).await {
            Ok(_) => changes.push(format!("Hugepages: {} allocated", req.hugepages_count)),
            Err(e) => errors.push(e),
        }
    }

    if req.set_swappiness {
        match kernel::set_swappiness(req.swappiness_value).await {
            Ok(_) => changes.push(format!("Swappiness → {}", req.swappiness_value)),
            Err(e) => errors.push(e),
        }
    }

    if req.set_io_scheduler {
        match io::set_scheduler(&req.io_scheduler).await {
            Ok(_) => changes.push(format!("I/O scheduler → {}", req.io_scheduler)),
            Err(e) => errors.push(e),
        }
    }

    if req.set_thp_mode {
        match kernel::set_thp_mode(&req.thp_mode).await {
            Ok(_) => changes.push(format!("THP mode → {}", req.thp_mode)),
            Err(e) => errors.push(e),
        }
    }

    if req.disable_numa_balancing {
        match kernel::disable_numa_balancing().await {
            Ok(_) => changes.push("NUMA balancing disabled".to_string()),
            Err(e) => errors.push(e),
        }
    }

    TuningResponse {
        applied: errors.is_empty(),
        changes_made: changes,
        errors,
    }
}
