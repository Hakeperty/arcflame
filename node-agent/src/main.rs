mod hardware;
mod network;
mod inference;
mod overclocking;
mod drivers;
mod tuning;

use clap::Parser;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;
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

    /// Start llama.cpp rpc-server alongside the node agent.
    #[arg(long, default_value = "false")]
    enable_rpc: bool,

    /// Port for llama.cpp rpc-server (default: grpc_port + 1000).
    #[arg(long)]
    rpc_port: Option<u16>,

    /// Path to llama-rpc-server binary.
    #[arg(long, default_value = "llama-rpc-server")]
    rpc_server_bin: PathBuf,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let args = Args::parse();

    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::new(&args.log_level))
        .init();

    let hostname_cached = hostname();
    let machine_id = machine_uid::get()
        .ok()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| hostname_cached.clone());
    let node_id = format!("{}-{}", machine_id, args.grpc_port);
    let hostname = hostname_cached;

    let hardware_report = hardware::collect().await?;
    tracing::info!(
        "Hardware detected on {}: {} cores, {} RAM",
        hostname,
        hardware_report.cpu.as_ref().map_or(0, |c| c.cores),
        hardware_report.memory.as_ref().map_or(0, |m| m.total_bytes),
    );

    // Determine effective RPC port
    let rpc_port: u16 = if args.enable_rpc {
        args.rpc_port.unwrap_or(args.grpc_port + 1000)
    } else {
        0
    };

    // Start llama.cpp rpc-server if requested
    if args.enable_rpc {
        let rpc = inference::rpc::RpcServer::new(rpc_port, args.rpc_server_bin.clone());
        match rpc.start().await {
            Ok(()) => tracing::info!("llama rpc-server running on port {}", rpc_port),
            Err(e) => tracing::warn!("Could not start rpc-server: {} (continuing without it)", e),
        }
        // Keep server alive for the process lifetime (leak is intentional here)
        Box::leak(Box::new(rpc));
    }

    let addr: SocketAddr = format!("0.0.0.0:{}", args.grpc_port).parse()?;

    let node_name = args.name.unwrap_or_else(|| hostname.clone());
    let node_id_clone = node_id.clone();
    let node_name_clone = node_name.clone();
    let grpc_port = args.grpc_port;

    // Start UDP discovery broadcaster (includes rpc_port)
    let shutdown = Arc::new(RwLock::new(false));
    network::discovery::start_broadcaster(
        node_id_clone,
        node_name_clone,
        grpc_port,
        rpc_port,
        shutdown,
    );

    // Register with orchestrator via HTTP
    if let Some(orchestrator_host) = args.orchestrator_host {
        let orch_addr = format!("{}:{}", orchestrator_host, args.orchestrator_port);
        let reg_url = format!("http://{}/api/nodes/register", orch_addr);
        let client = reqwest::Client::new();
        let payload = serde_json::json!({
            "node_id": node_id,
            "name": node_name,
            "grpc_port": grpc_port,
            "rpc_port": rpc_port,
            "version": env!("CARGO_PKG_VERSION"),
            "os": std::env::consts::OS,
        });
        match client.post(&reg_url).json(&payload).send().await {
            Ok(resp) => {
                if resp.status().is_success() {
                    tracing::info!("Registered with orchestrator at {}", orch_addr);
                } else {
                    tracing::warn!("Orchestrator registration failed: {}", resp.status());
                }
            }
            Err(e) => {
                tracing::warn!("Could not reach orchestrator at {}: {}", orch_addr, e);
            }
        }
    }

    let node_svc = network::grpc_server::ArcFlareNodeService::new(
        node_id,
        node_name,
        hardware_report,
        addr,
        rpc_port,
    );

    tracing::info!("Node agent starting on {}", addr);

    network::grpc_server::serve(node_svc, addr).await
}

fn hostname() -> String {
    std::fs::read_to_string("/etc/hostname")
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|_| "unknown".to_string())
}
