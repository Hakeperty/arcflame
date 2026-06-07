mod hardware;
mod network;
#[cfg(feature = "inference")]
mod inference;
mod overclocking;
mod drivers;
mod tuning;

use clap::Parser;
use std::net::SocketAddr;
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(name = "arcflare-node", about = "ArcFlare cluster node agent")]
struct Args {
    #[arg(short, long, default_value = "9001")]
    grpc_port: u16,

    #[arg(short, long)]
    name: Option<String>,

    #[arg(short, long, default_value = "8000")]
    orchestrator_port: u16,

    #[arg(short, long)]
    orchestrator_host: Option<String>,

    #[arg(long, default_value = "info")]
    log_level: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let args = Args::parse();

    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::new(&args.log_level))
        .init();

    let node_id = machine_uid::get().unwrap_or_else(|_| uuid::Uuid::new_v4().to_string());
    let hostname = hostname();

    let hardware_report = hardware::collect().await?;
    tracing::info!(
        "Hardware detected on {}: {} cores, {} RAM",
        hostname,
        hardware_report.cpu.as_ref().map_or(0, |c| c.cores),
        hardware_report.memory.as_ref().map_or(0, |m| m.total_bytes),
    );

    let addr: SocketAddr = format!("0.0.0.0:{}", args.grpc_port).parse()?;

    let node_name = args.name.unwrap_or_else(|| hostname.clone());

    let node_svc = network::grpc_server::ArcFlareNodeService::new(
        node_id,
        node_name,
        hardware_report,
        addr,
    );

    tracing::info!("Node agent starting on {}", addr);

    network::grpc_server::serve(node_svc, addr).await
}

fn hostname() -> String {
    std::fs::read_to_string("/etc/hostname")
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|_| "unknown".to_string())
}
