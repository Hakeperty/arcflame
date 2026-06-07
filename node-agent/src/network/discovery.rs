#![allow(dead_code)]
use std::net::UdpSocket;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn};

const DISCOVERY_PORT: u16 = 5678;
const DISCOVERY_INTERVAL: std::time::Duration = std::time::Duration::from_secs(5);

#[derive(Clone, serde::Serialize, serde::Deserialize, Debug)]
pub struct DiscoveryMessage {
    pub node_id: String,
    pub node_name: String,
    pub grpc_port: u16,
    pub version: String,
    pub os: String,
}

pub fn start_broadcaster(
    node_id: String,
    node_name: String,
    grpc_port: u16,
    shutdown: Arc<RwLock<bool>>,
) {
    let socket = match UdpSocket::bind("0.0.0.0:0") {
        Ok(s) => {
            s.set_broadcast(true).ok();
            s
        }
        Err(e) => {
            warn!("Failed to bind discovery socket: {}", e);
            return;
        }
    };

    let msg = DiscoveryMessage {
        node_id,
        node_name,
        grpc_port,
        version: env!("CARGO_PKG_VERSION").to_string(),
        os: std::env::consts::OS.to_string(),
    };

    let payload = match serde_json::to_vec(&msg) {
        Ok(p) => p,
        Err(e) => {
            warn!("Failed to serialize discovery message: {}", e);
            return;
        }
    };

    info!("Starting UDP discovery broadcaster on port {}", DISCOVERY_PORT);

    tokio::spawn(async move {
        loop {
            if *shutdown.read().await {
                break;
            }

            if let Err(e) = socket.send_to(&payload, format!("255.255.255.255:{}", DISCOVERY_PORT)) {
                warn!("Discovery broadcast failed: {}", e);
            }

            tokio::time::sleep(DISCOVERY_INTERVAL).await;
        }
    });
}
