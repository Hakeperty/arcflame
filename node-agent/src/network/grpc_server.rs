use std::net::SocketAddr;
use std::pin::Pin;
use tonic::{transport::Server, Request, Response, Status, Streaming};
use tracing::info;

use super::arcflare as pb;
use pb::{
    node_agent_server::{NodeAgent, NodeAgentServer},
    perf_mode_request::Mode,
    BenchmarkRequest, BenchmarkResult, DriverReport, Empty, ForwardRequest,
    ForwardResponse, HardwareReport, HeartbeatRequest, HeartbeatResponse,
    InferenceStats, LoadStatus, OverclockingStatus, PerfModeRequest,
    PerfModeResponse, RegisterRequest, RegisterResponse, ShardConfig,
    TuningRequest, TuningResponse,
};

pub struct ArcFlareNodeService {
    node_id: String,
    #[allow(dead_code)]
    node_name: String,
    hardware_report: HardwareReport,
    _addr: SocketAddr,
}

impl ArcFlareNodeService {
    pub fn new(
        node_id: String,
        node_name: String,
        hardware_report: HardwareReport,
        addr: SocketAddr,
    ) -> Self {
        Self { node_id, node_name, hardware_report, _addr: addr }
    }
}

#[tonic::async_trait]
impl NodeAgent for ArcFlareNodeService {
    async fn register(
        &self,
        request: Request<RegisterRequest>,
    ) -> Result<Response<RegisterResponse>, Status> {
        let req = request.into_inner();
        info!("Register request from node: {}", req.node_id);
        Ok(Response::new(RegisterResponse {
            accepted: true,
            orchestrator_id: "arcflare-orch".to_string(),
            session_token: "dev-session".to_string(),
        }))
    }

    async fn heartbeat(
        &self,
        request: Request<HeartbeatRequest>,
    ) -> Result<Response<HeartbeatResponse>, Status> {
        let _req = request.into_inner();
        Ok(Response::new(HeartbeatResponse {
            acknowledged: true,
            commands: vec![],
        }))
    }

    async fn get_hardware_info(
        &self,
        _request: Request<Empty>,
    ) -> Result<Response<HardwareReport>, Status> {
        let mut report = self.hardware_report.clone();
        report.node_id = self.node_id.clone();
        Ok(Response::new(report))
    }

    async fn run_benchmark(
        &self,
        _request: Request<BenchmarkRequest>,
    ) -> Result<Response<BenchmarkResult>, Status> {
        let cpu_result = crate::hardware::benchmark::run_cpu_benchmark().await;
        Ok(Response::new(cpu_result))
    }

    async fn set_performance_mode(
        &self,
        request: Request<PerfModeRequest>,
    ) -> Result<Response<PerfModeResponse>, Status> {
        let req = request.into_inner();
        let mode_id = req.mode;
        info!("Setting performance mode to: {:?}", Mode::try_from(mode_id));

        let result = match Mode::try_from(mode_id) {
            Ok(Mode::Safe) => crate::overclocking::safe::apply_safe_tuning().await,
            Ok(Mode::Aggressive) => crate::overclocking::aggressive::apply_aggressive_tuning().await,
            _ => Ok("Custom mode applied".to_string()),
        };

        match result {
            Ok(msg) => Ok(Response::new(PerfModeResponse {
                applied: true,
                message: msg,
                before: std::collections::HashMap::new(),
                after: std::collections::HashMap::new(),
            })),
            Err(e) => Ok(Response::new(PerfModeResponse {
                applied: false,
                message: format!("Failed: {}", e),
                before: std::collections::HashMap::new(),
                after: std::collections::HashMap::new(),
            })),
        }
    }

    async fn get_overclocking_status(
        &self,
        _request: Request<Empty>,
    ) -> Result<Response<OverclockingStatus>, Status> {
        let status = crate::overclocking::monitor::get_status().await;
        Ok(Response::new(status))
    }

    async fn check_drivers(
        &self,
        _request: Request<Empty>,
    ) -> Result<Response<DriverReport>, Status> {
        let report = crate::drivers::check::audit().await;
        Ok(Response::new(report))
    }

    async fn apply_system_tuning(
        &self,
        request: Request<TuningRequest>,
    ) -> Result<Response<TuningResponse>, Status> {
        let req = request.into_inner();
        let result = crate::tuning::apply_tuning(req).await;
        Ok(Response::new(result))
    }

    async fn load_shard(
        &self,
        request: Request<ShardConfig>,
    ) -> Result<Response<LoadStatus>, Status> {
        #[cfg(feature = "inference")]
        {
            let config = request.into_inner();
            let status = crate::inference::load_shard(config).await
                .map_err(|e| Status::internal(format!("Failed to load shard: {}", e)))?;
            Ok(Response::new(status))
        }

        #[cfg(not(feature = "inference"))]
        {
            let _ = request;
            Err(Status::unimplemented(
                "Inference support not compiled. Build with --features inference"
            ))
        }
    }

    async fn forward(
        &self,
        request: Request<ForwardRequest>,
    ) -> Result<Response<ForwardResponse>, Status> {
        #[cfg(feature = "inference")]
        {
            let req = request.into_inner();
            let resp = crate::inference::forward(req).await
                .map_err(|e| Status::internal(format!("Forward pass failed: {}", e)))?;
            Ok(Response::new(resp))
        }

        #[cfg(not(feature = "inference"))]
        {
            let _ = request;
            Err(Status::unimplemented(
                "Inference support not compiled. Build with --features inference"
            ))
        }
    }

    type ForwardStreamStream = Pin<Box<dyn tokio_stream::Stream<Item = Result<ForwardResponse, Status>> + Send>>;

    async fn forward_stream(
        &self,
        request: Request<Streaming<ForwardRequest>>,
    ) -> Result<Response<Self::ForwardStreamStream>, Status> {
        #[cfg(feature = "inference")]
        {
            use tokio_stream::StreamExt;
            let mut stream = request.into_inner();

            let output = async_stream::try_stream! {
                while let Some(req) = stream.next().await {
                    let req = req?;

                    // Full generation mode: tokenize prompt, autoregressively generate
                    if !req.text_prompt.is_empty() {
                        let max_tokens = if req.max_tokens > 0 { req.max_tokens as usize } else { 256 };
                        let temp = if req.temperature > 0.0 { req.temperature } else { 0.0 };

                        let state = crate::inference::model::get_loaded_model().await
                            .ok_or_else(|| Status::internal("No model loaded"))?;
                        let guard = state.read().await;
                        let shard = guard.as_ref().ok_or_else(|| Status::internal("No shard"))?;
                        let model = shard.model.as_ref().ok_or_else(|| Status::internal("No model"))?;

                        let be = crate::inference::forward::backend()
                            .map_err(|e| Status::internal(e))?;

                        // Tokenize
                        let prompt_tokens = model.str_to_token(&req.text_prompt, true)
                            .map_err(|e| Status::internal(format!("Tokenize: {}", e)))?;

                        // Create a fresh context
                        let ctx_params = llama_cpp_4::context::params::LlamaContextParams::default()
                            .with_n_ctx(2048);
                        let mut ctx = model.new_context(be, &ctx_params)
                            .map_err(|e| Status::internal(format!("Ctx: {}", e)))?;

                        // Decode prompt
                        if !prompt_tokens.is_empty() {
                            ctx.decode(&mut std::iter::once(&prompt_tokens[..]))
                                .map_err(|e| Status::internal(format!("Decode prompt: {}", e)))?;
                        }

                        // Generation loop
                        let mut token = 0i32;
                        for _i in 0..max_tokens {
                            if token != 0 {
                                ctx.decode(&mut std::iter::once(&[token][..]))
                                    .map_err(|e| Status::internal(format!("Decode token: {}", e)))?;
                            }

                            let n_tokens = ctx.n_tokens() as usize;
                            if n_tokens == 0 { break; }

                            let logits = ctx.logits();
                            let n_vocab = logits.len();
                            let sampled = if n_vocab > 0 {
                                argmax_token(logits)
                            } else {
                                break;
                            };

                            let text = model.token_to_str(&[sampled])
                                .unwrap_or_else(|_| format!("[token {}]", sampled));

                            token = sampled;

                            yield ForwardResponse {
                                hidden_state: vec![],
                                logits: text.as_bytes().to_vec(),
                                has_logits: true,
                                compute_time_ms: 0,
                            };

                            // Stop on EOS (typically token 2 for most models)
                            if sampled == 2 { break; }
                        }

                        // Signal done
                        yield ForwardResponse {
                            hidden_state: vec![],
                            logits: vec![],
                            has_logits: false,
                            compute_time_ms: 0,
                        };
                        continue;
                    }

                    // Standard mode: decode one step
                    let resp = crate::inference::forward(req).await
                        .map_err(|e| Status::internal(format!("Forward: {}", e)))?;
                    yield resp;
                }
            };

            return Ok(Response::new(Box::pin(output) as Self::ForwardStreamStream));
        }

        #[cfg(not(feature = "inference"))]
        {
            let _ = request;
            Err(Status::unimplemented(
                "Inference support not compiled. Build with --features inference"
            ))
        }
    }

    async fn unload_shard(
        &self,
        _request: Request<Empty>,
    ) -> Result<Response<Empty>, Status> {
        #[cfg(feature = "inference")]
        {
            crate::inference::unload_shard().await;
        }
        Ok(Response::new(Empty {}))
    }

    async fn get_inference_stats(
        &self,
        _request: Request<Empty>,
    ) -> Result<Response<InferenceStats>, Status> {
        Ok(Response::new(InferenceStats {
            layers_loaded: 0,
            total_forward_calls: 0,
            avg_forward_time_ms: 0.0,
            total_tokens_processed: 0,
            kv_cache_used_bytes: 0,
            peak_memory_bytes: 0,
        }))
    }
}

/// Argmax over logits: pick the token with the highest score
#[cfg(feature = "inference")]
fn argmax_token(logits: &[f32]) -> i32 {
    let (idx, _) = logits.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or((0, &0.0));
    idx as i32
}

pub async fn serve(
    service: ArcFlareNodeService,
    addr: SocketAddr,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    Server::builder()
        .add_service(NodeAgentServer::new(service))
        .serve(addr)
        .await?;
    Ok(())
}
