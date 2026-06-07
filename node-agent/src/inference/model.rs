use crate::network::arcflare as pb;
use pb::{LoadStatus, ShardConfig};
use std::sync::Arc;
use tokio::sync::RwLock;

#[cfg(feature = "inference")]
use llama_cpp_4::{llama_backend::LlamaBackend, model::LlamaModel, model::params::LlamaModelParams};

// Global model state
static CURRENT_SHARD: once_cell::sync::OnceCell<Arc<RwLock<Option<ShardState>>>> =
    once_cell::sync::OnceCell::new();

pub struct ShardState {
    pub model_name: String,
    pub first_layer: i32,
    pub num_layers: i32,
    pub has_lm_head: bool,
    #[cfg(feature = "inference")]
    pub model: Option<LlamaModel>,
}

pub async fn get_loaded_model() -> Option<Arc<RwLock<Option<ShardState>>>> {
    CURRENT_SHARD.get().map(|c| c.clone())
}

pub async fn load_shard_model(config: ShardConfig) -> Result<LoadStatus, String> {
    let start = std::time::Instant::now();

    #[cfg(feature = "inference")]
    {
        let backend = LlamaBackend::init()
            .map_err(|e| format!("Failed to init backend: {}", e))?;
        let params = LlamaModelParams::default()
            .with_n_gpu_layers(999);
        let model = LlamaModel::load_from_file(
            &backend,
            &config.gguf_path,
            &params,
        ).map_err(|e| format!("Failed to load model: {}", e))?;

        let state = ShardState {
            model_name: config.model_name.clone(),
            first_layer: config.first_layer,
            num_layers: config.num_layers,
            has_lm_head: config.has_lm_head,
            model: Some(model),
        };

        let cell = CURRENT_SHARD.get_or_init(|| Arc::new(RwLock::new(None)));
        *cell.write().await = Some(state);

        let elapsed = start.elapsed();
        Ok(LoadStatus {
            loaded: true,
            memory_used_bytes: 0,
            layers_loaded: config.num_layers,
            load_time_ms: elapsed.as_millis() as i64,
            error: String::new(),
        })
    }

    #[cfg(not(feature = "inference"))]
    {
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        let elapsed = start.elapsed();
        Ok(LoadStatus {
            loaded: true,
            memory_used_bytes: 512 * 1024 * 1024,
            layers_loaded: config.num_layers,
            load_time_ms: elapsed.as_millis() as i64,
            error: String::new(),
        })
    }
}

pub async fn unload_current_shard() {
    if let Some(cell) = CURRENT_SHARD.get() {
        *cell.write().await = None;
    }
}
