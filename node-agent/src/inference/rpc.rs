use std::path::PathBuf;
use std::sync::Arc;
use tokio::process::Child;
use tokio::sync::Mutex;
use tracing::{info, warn};

pub struct RpcServer {
    port: u16,
    bin: PathBuf,
    process: Arc<Mutex<Option<Child>>>,
}

impl RpcServer {
    pub fn new(port: u16, bin: PathBuf) -> Self {
        Self {
            port,
            bin,
            process: Arc::new(Mutex::new(None)),
        }
    }

    /// Start `llama-rpc-server -p <port>` as a child process.
    pub async fn start(&self) -> Result<(), String> {
        let mut guard = self.process.lock().await;
        if guard.is_some() {
            return Ok(()); // already running
        }

        let child = tokio::process::Command::new(&self.bin)
            .arg("-p")
            .arg(self.port.to_string())
            .spawn()
            .map_err(|e| format!("Failed to spawn rpc-server ({}): {}", self.bin.display(), e))?;

        info!("llama rpc-server started on port {}", self.port);
        *guard = Some(child);
        Ok(())
    }

    /// Kill the rpc-server if running.
    pub async fn stop(&self) {
        let mut guard = self.process.lock().await;
        if let Some(mut child) = guard.take() {
            if let Err(e) = child.kill().await {
                warn!("Failed to kill rpc-server: {}", e);
            }
        }
    }

    pub fn port(&self) -> u16 {
        self.port
    }

    pub async fn is_running(&self) -> bool {
        self.process.lock().await.is_some()
    }
}
