pub mod model;
pub mod shard;
pub mod forward;

use crate::network::arcflare as pb;
use pb::{ForwardRequest, ForwardResponse, LoadStatus, ShardConfig};

pub async fn load_shard(config: ShardConfig) -> Result<LoadStatus, String> {
    model::load_shard_model(config).await
}

pub async fn forward(request: ForwardRequest) -> Result<ForwardResponse, String> {
    forward::run_forward_pass(request).await
}

pub async fn unload_shard() {
    model::unload_current_shard().await;
}
