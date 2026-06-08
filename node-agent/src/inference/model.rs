use crate::network::arcflare as pb;
use pb::{LoadStatus, ShardConfig};
use std::sync::Arc;
use tokio::sync::RwLock;

#[cfg(feature = "inference")]
use llama_cpp_4::model::params::LlamaModelParams;
#[cfg(feature = "inference")]
use llama_cpp_4::model::LlamaModel;

static CURRENT_SHARD: once_cell::sync::OnceCell<Arc<RwLock<Option<ShardState>>>> =
    once_cell::sync::OnceCell::new();

pub struct ShardState {
    #[allow(dead_code)]
    pub model_name: String,
    #[allow(dead_code)]
    pub first_layer: i32,
    #[allow(dead_code)]
    pub num_layers: i32,
    #[allow(dead_code)]
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
        // Backend must be initialized first (from forward::backend)
        let _ = crate::inference::forward::backend()?;

        // Drop params before any await to avoid Send issues
        let model = {
            let params = LlamaModelParams::default()
                .with_n_gpu_layers(999);
            LlamaModel::load_from_file(
                crate::inference::forward::backend().map_err(|e| e.to_string())?,
                &config.gguf_path,
                &params,
            ).map_err(|e| format!("Failed to load model: {}", e))?
        };

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
        let _ = config;
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
