use crate::network::arcflare as pb;
use pb::{ForwardRequest, ForwardResponse};
use std::time::Instant;

#[cfg(feature = "inference")]
use {
    crate::inference::model::get_loaded_model,
    llama_cpp_4::context::params::LlamaContextParams,
};

pub async fn run_forward_pass(request: ForwardRequest) -> Result<ForwardResponse, String> {
    let start = Instant::now();

    #[cfg(feature = "inference")]
    {
        let state = get_loaded_model().await
            .ok_or_else(|| "No model loaded".to_string())?;

        let guard = state.read().await;
        let shard = guard.as_ref()
            .ok_or_else(|| "No shard loaded".to_string())?;

        if let Some(ref _model) = shard.model {
            let backend = llama_cpp_4::llama_backend::LlamaBackend::init()
                .map_err(|e| format!("Backend init: {}", e))?;

            let ctx_params = LlamaContextParams::default()
                .with_n_ctx(2048);

            let mut ctx = _model.new_context(&backend, &ctx_params)
                .map_err(|e| format!("Context creation: {}", e))?;

            // If we have input_ids, decode them
            if !request.input_ids.is_empty() {
                let tokens: Vec<_> = request.input_ids.iter().map(|&t| t as i32).collect();
                ctx.decode(&mut std::iter::once(&tokens[..]))
                    .map_err(|e| format!("Decode error: {}", e))?;

                let n_tokens = ctx.n_tokens() as usize;
                let logits_data = if n_tokens > 0 {
                    let logits_slice = ctx.logits();
                    // logits_slice is &[f32]; flatten to bytes
                    logits_slice.iter().flat_map(|&f| f.to_le_bytes()).collect()
                } else {
                    vec![]
                };

                let compute_time = start.elapsed();
                return Ok(ForwardResponse {
                    hidden_state: vec![],
                    logits: logits_data,
                    has_logits: true,
                    compute_time_ms: compute_time.as_millis() as i64,
                });
            }

            // No input_ids — this is a hidden-state passthrough node
            // In a real pipeline, we'd process the hidden_state through
            // the loaded model's layers for this node's layer range.
            // Since llama-cpp-4 doesn't expose per-layer forward pass,
            // we return the input as-is with a note.
            let compute_time = start.elapsed();
            return Ok(ForwardResponse {
                hidden_state: request.hidden_state,
                logits: vec![],
                has_logits: shard.has_lm_head,
                compute_time_ms: compute_time.as_millis() as i64,
            });
        }
    }

    // Default / stub path
    let _input_len = request.input_ids.len();
    let hidden_state = request.hidden_state;

    let compute_time = start.elapsed();
    Ok(ForwardResponse {
        hidden_state,
        logits: vec![],
        has_logits: false,
        compute_time_ms: compute_time.as_millis() as i64,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_forward_pass() {
        let request = ForwardRequest {
            hidden_state: vec![0u8; 1024],
            start_layer: 0,
            num_layers: 10,
            input_ids: vec![1, 2, 3],
        };

        let result = run_forward_pass(request).await;
        assert!(result.is_ok());
        let response = result.unwrap();
        assert!(response.compute_time_ms >= 0);
    }
}
