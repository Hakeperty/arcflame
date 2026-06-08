pub mod rpc;

#[cfg(feature = "inference")]
pub mod model;
#[cfg(feature = "inference")]
pub mod shard;
#[cfg(feature = "inference")]
pub mod forward;

#[cfg(feature = "inference")]
use crate::network::arcflare as pb;
#[cfg(feature = "inference")]
use pb::{ForwardRequest, ForwardResponse, LoadStatus, ShardConfig};

#[cfg(feature = "inference")]
pub async fn load_shard(config: ShardConfig) -> Result<LoadStatus, String> {
    model::load_shard_model(config).await
}

#[cfg(feature = "inference")]
pub async fn forward(request: ForwardRequest) -> Result<ForwardResponse, String> {
    forward::run_forward_pass(request).await
}

#[cfg(feature = "inference")]
pub async fn unload_shard() {
    model::unload_current_shard().await;
}
