use crate::network::arcflare as pb;
use pb::{ForwardRequest, ForwardResponse};
use std::time::Instant;

pub async fn run_forward_pass(request: ForwardRequest) -> Result<ForwardResponse, String> {
    let start = Instant::now();

    // The actual forward pass happens in llama-cpp-4
    // For Phase 1, we simulate the pipeline
    //
    // In production:
    // 1. Decode the incoming hidden_state tensor
    // 2. Run through llama-cpp-4's layers for our shard
    // 3. If we have the LM head, return logits
    // 4. Otherwise, return the hidden state for the next node

    let _input_len = request.input_ids.len();
    let hidden_state = request.hidden_state;

    let compute_time = start.elapsed();

    Ok(ForwardResponse {
        hidden_state, // Pass through (simulated)
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
