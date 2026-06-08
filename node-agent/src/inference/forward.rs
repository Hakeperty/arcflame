use crate::network::arcflare as pb;
use pb::{ForwardRequest, ForwardResponse};
use std::time::Instant;

#[cfg(feature = "inference")]
use {
    crate::inference::model::get_loaded_model,
    llama_cpp_4::context::params::LlamaContextParams,
    llama_cpp_4::llama_backend::LlamaBackend,
    llama_cpp_4::context::LlamaContext,
    std::sync::Mutex,
};

#[cfg(feature = "inference")]
static BACKEND: once_cell::sync::OnceCell<&'static LlamaBackend> =
    once_cell::sync::OnceCell::new();

#[cfg(feature = "inference")]
pub fn backend() -> Result<&'static LlamaBackend, String> {
    BACKEND.get_or_try_init(|| {
        let b = LlamaBackend::init().map_err(|e| format!("Backend init: {}", e))?;
        Ok(Box::leak(Box::new(b)))
    }).copied()
}

#[cfg(feature = "inference")]
static CTX: once_cell::sync::OnceCell<Mutex<Option<LlamaContext<'static>>>> =
    once_cell::sync::OnceCell::new();

pub async fn run_forward_pass(request: ForwardRequest) -> Result<ForwardResponse, String> {
    let start = Instant::now();

    #[cfg(feature = "inference")]
    {
        let be = backend()?;
        let state = get_loaded_model().await
            .ok_or_else(|| "No model loaded".to_string())?;
        let guard = state.read().await;
        let shard = guard.as_ref()
            .ok_or_else(|| "No shard loaded".to_string())?;
        let model = shard.model.as_ref()
            .ok_or_else(|| "No model instance".to_string())?;

        let cell = CTX.get_or_init(|| Mutex::new(None));
        let mut ctx_guard = cell.lock().map_err(|e| format!("Lock: {}", e))?;

        let ctx = ctx_guard.get_or_insert_with(|| {
            let params = LlamaContextParams::default()
                .with_n_ctx(2048);
            model.new_context(be, &params)
                .expect("Failed to create context")
        });

        // Reset context if requested
        if request.clear_context {
            *ctx_guard = None;
        }

        // Tokenize text prompt if provided
        let resolved_ids: Vec<i32> = if !request.text_prompt.is_empty() {
            model.str_to_token(&request.text_prompt, true)
                .map_err(|e| format!("Tokenize error: {}", e))?
        } else {
            request.input_ids.iter().map(|&t| t as i32).collect()
        };

        if !resolved_ids.is_empty() {
            // Ensure context exists (re-create if was cleared)
            let ctx = ctx_guard.get_or_insert_with(|| {
                let params = LlamaContextParams::default()
                    .with_n_ctx(2048);
                model.new_context(be, &params)
                    .expect("Failed to create context")
            });

            ctx.decode(&mut std::iter::once(&resolved_ids[..]))
                .map_err(|e| format!("Decode error: {}", e))?;
            ctx.decode(&mut std::iter::once(&tokens[..]))
                .map_err(|e| format!("Decode error: {}", e))?;

            let n_tokens = ctx.n_tokens() as usize;
            let logits_data = if n_tokens > 0 {
                ctx.logits().iter().flat_map(|&f| f.to_le_bytes()).collect()
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

        // No input_ids — passthrough (hidden state node)
        let compute_time = start.elapsed();
        return Ok(ForwardResponse {
            hidden_state: request.hidden_state,
            logits: vec![],
            has_logits: shard.has_lm_head,
            compute_time_ms: compute_time.as_millis() as i64,
        });
    }

    // Stub path
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
