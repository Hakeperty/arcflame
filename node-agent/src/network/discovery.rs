#![allow(dead_code)]
use std::sync::Arc;
use tokio::net::UdpSocket;
use tokio::sync::RwLock;
use tracing::{info, warn};

const DISCOVERY_PORT: u16 = 5678;
const DISCOVERY_INTERVAL: std::time::Duration = std::time::Duration::from_secs(5);

#[derive(Clone, serde::Serialize, serde::Deserialize, Debug)]
pub struct DiscoveryMessage {
    pub node_id: String,
    pub node_name: String,
    pub grpc_port: u16,
    pub rpc_port: u16,  // 0 = not running
    pub version: String,
    pub os: String,
}

pub fn start_broadcaster(
    node_id: String,
    node_name: String,
    grpc_port: u16,
    rpc_port: u16,
    shutdown: Arc<RwLock<bool>>,
) {
    let msg = DiscoveryMessage {
        node_id,
        node_name,
        grpc_port,
        rpc_port,
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
        // Bind inside the task using tokio's async socket so send_to never
        // blocks the runtime (was std::net::UdpSocket — a blocking call).
        let socket = match UdpSocket::bind("0.0.0.0:0").await {
            Ok(s) => s,
            Err(e) => {
                warn!("Failed to bind discovery socket: {}", e);
                return;
            }
        };
        // SO_BROADCAST is required for 255.255.255.255 sends; a failure here
        // would make every send fail, so abort the broadcaster instead of
        // silently swallowing it (was `.ok()`).
        if let Err(e) = socket.set_broadcast(true) {
            warn!("Failed to enable SO_BROADCAST on discovery socket: {}", e);
            return;
        }

        let dest = format!("255.255.255.255:{}", DISCOVERY_PORT);
        loop {
            if *shutdown.read().await {
                break;
            }
            if let Err(e) = socket.send_to(&payload, &dest).await {
                warn!("Discovery broadcast failed: {}", e);
            }
            // Wake promptly on shutdown instead of waiting the full interval.
            tokio::select! {
                _ = tokio::time::sleep(DISCOVERY_INTERVAL) => {}
                _ = async {
                    // poll the flag a few times within the interval
                    for _ in 0..(DISCOVERY_INTERVAL.as_secs().max(1)) {
                        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                        if *shutdown.read().await { break; }
                    }
                } => {}
            }
        }
        info!("Discovery broadcaster stopped");
    });
}
