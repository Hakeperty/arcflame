use crate::network::arcflare as pb;
use pb::{ForwardRequest, ForwardResponse};
use std::time::Instant;

#[cfg(feature = "inference")]
use {
    crate::inference::model::get_loaded_model,
    llama_cpp_4::context::params::LlamaContextParams,
    llama_cpp_4::llama_backend::LlamaBackend,
    llama_cpp_4::llama_batch::LlamaBatch,
    llama_cpp_4::model::AddBos,
    std::num::NonZero,
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

        // Resolve tokens
        let llama_tokens: Vec<llama_cpp_4::token::LlamaToken> = if !request.text_prompt.is_empty() {
            model.str_to_token(&request.text_prompt, AddBos::Always)
                .map_err(|e| format!("Tokenize error: {}", e))?
        } else {
            request.input_ids.iter().map(|&t| llama_cpp_4::token::LlamaToken(t)).collect()
        };

        // All inference ops are synchronous — no await after this point
        let n_ctx = NonZero::new(2048u32).ok_or("Invalid ctx size")?;
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(Some(n_ctx));

        let mut ctx = model.new_context(be, ctx_params)
            .map_err(|e| format!("Context creation: {}", e))?;

        let mut batch = LlamaBatch::new(llama_tokens.len(), 1);
        batch.add_sequence(&llama_tokens, 0, true)
            .map_err(|e| format!("Batch add: {}", e))?;

        ctx.decode(&mut batch)
            .map_err(|e| format!("Decode error: {}", e))?;

        let logits_data: Vec<u8> = ctx.get_logits()
            .iter()
            .flat_map(|&f| f.to_le_bytes())
            .collect();

        // Drop ctx, batch, model reference, shard guard before returning
        drop(ctx);
        drop(batch);
        drop(guard);
        drop(state);

        let compute_time = start.elapsed();
        return Ok(ForwardResponse {
            hidden_state: vec![],
            logits: logits_data,
            has_logits: true,
            compute_time_ms: compute_time.as_millis() as i64,
        });
    }

    // Stub path (also reached when inference feature is disabled)
    #[cfg(not(feature = "inference"))]
    {
        let _input_len = request.input_ids.len();
        let hidden_state = request.hidden_state;
        let compute_time = start.elapsed();
        return Ok(ForwardResponse {
            hidden_state,
            logits: vec![],
            has_logits: false,
            compute_time_ms: compute_time.as_millis() as i64,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(not(feature = "inference"))]
    #[tokio::test]
    async fn test_forward_pass_stub() {
        let request = ForwardRequest {
            hidden_state: vec![0u8; 1024],
            start_layer: 0,
            num_layers: 10,
            input_ids: vec![1, 2, 3],
            text_prompt: String::new(),
            clear_context: false,
            max_tokens: 0,
            temperature: 0.0,
        };
        let result = run_forward_pass(request).await;
        assert!(result.is_ok());
        let response = result.unwrap();
        assert!(response.compute_time_ms >= 0);
    }
}
