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
    PerfModeResponse, RegisterRequest, RegisterResponse, RpcEndpoint, ShardConfig,
    TuningRequest, TuningResponse,
};

pub struct ArcFlareNodeService {
    node_id: String,
    #[allow(dead_code)]
    node_name: String,
    hardware_report: HardwareReport,
    _addr: SocketAddr,
    rpc_port: u16,
    // Opt-in gates: the gRPC server is unauthenticated, so system- and
    // hardware-modifying RPCs are refused unless the operator explicitly
    // enabled them at launch. Prevents a LAN peer from overclocking/undervolting
    // or retuning the kernel on this machine.
    allow_tuning: bool,
    allow_aggressive: bool,
}

impl ArcFlareNodeService {
    pub fn new(
        node_id: String,
        node_name: String,
        hardware_report: HardwareReport,
        addr: SocketAddr,
        rpc_port: u16,
        allow_tuning: bool,
        allow_aggressive: bool,
    ) -> Self {
        Self {
            node_id,
            node_name,
            hardware_report,
            _addr: addr,
            rpc_port,
            allow_tuning,
            allow_aggressive,
        }
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

        // Gate hardware/system changes behind launch-time opt-in (unauthenticated RPC).
        match Mode::try_from(mode_id) {
            Ok(Mode::Safe) if !self.allow_tuning => {
                return Err(Status::permission_denied(
                    "safe tuning disabled; start the agent with --allow-tuning",
                ));
            }
            Ok(Mode::Aggressive) if !self.allow_aggressive => {
                return Err(Status::permission_denied(
                    "aggressive overclock/undervolt disabled; start the agent with --allow-aggressive",
                ));
            }
            _ => {}
        }

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
        if !self.allow_tuning {
            return Err(Status::permission_denied(
                "system tuning disabled; start the agent with --allow-tuning",
            ));
        }
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
            use tokio_stream::wrappers::ReceiverStream;
            let mut stream = request.into_inner();

            let output = async_stream::try_stream! {
                while let Some(req) = stream.next().await {
                    let req = req?;

                    if !req.text_prompt.is_empty() {
                        let prompt_text = req.text_prompt.clone();
                        let max_tokens = if req.max_tokens > 0 { req.max_tokens as usize } else { 256 };

                        let (tx, rx) = tokio::sync::mpsc::channel::<
                            Result<ForwardResponse, Status>
                        >(32);

                        tokio::task::spawn_blocking(move || {
                            use llama_cpp_4::context::params::LlamaContextParams;
                            use llama_cpp_4::llama_batch::LlamaBatch;
                            use llama_cpp_4::model::params::LlamaModelParams;
                            use llama_cpp_4::model::{AddBos, LlamaModel, Special};
                            use std::num::NonZero;

                            // Use the shared process-global backend. Calling
                            // LlamaBackend::init() here re-initializes the singleton and
                            // errors after the first request (and races concurrent ones).
                            let be = match crate::inference::forward::backend() {
                                Ok(b) => b,
                                Err(e) => {
                                    let _ = tx.blocking_send(Err(Status::internal(
                                        format!("Backend init: {}", e)
                                    )));
                                    return;
                                }
                            };

                            // TODO(perf #3): reuse the already-loaded shard model from
                            // get_loaded_model() instead of reloading the GGUF per request.
                            let model_path = std::env::var("ARCFLARE_MODEL_PATH")
                                .unwrap_or_else(|_| {
                                    "/models/qwen2.5-0.5b-instruct-q4_k_m.gguf".to_string()
                                });

                            let model_params = LlamaModelParams::default()
                                .with_n_gpu_layers(999);
                            let model = match LlamaModel::load_from_file(be, &model_path, &model_params)
                            {
                                Ok(m) => m,
                                Err(e) => {
                                    let _ = tx.blocking_send(Err(Status::internal(
                                        format!("Load model: {}", e)
                                    )));
                                    return;
                                }
                            };

                            let prompt_tokens = match model.str_to_token(&prompt_text, AddBos::Always)
                            {
                                Ok(t) => t,
                                Err(e) => {
                                    let _ = tx.blocking_send(Err(Status::internal(
                                        format!("Tokenize: {}", e)
                                    )));
                                    return;
                                }
                            };

                            let n_ctx = match NonZero::new(2048u32) {
                                Some(n) => n,
                                None => {
                                    let _ = tx.blocking_send(Err(Status::internal("Invalid ctx")));
                                    return;
                                }
                            };
                            let ctx_params = LlamaContextParams::default()
                                .with_n_ctx(Some(n_ctx));
                            let mut ctx = match model.new_context(be, ctx_params) {
                                Ok(c) => c,
                                Err(e) => {
                                    let _ = tx.blocking_send(Err(Status::internal(
                                        format!("Ctx: {}", e)
                                    )));
                                    return;
                                }
                            };

                            // Batch and decode the prompt
                            let mut batch = LlamaBatch::new(prompt_tokens.len(), 1);
                            if let Err(e) = batch.add_sequence(&prompt_tokens, 0, true) {
                                let _ = tx.blocking_send(Err(Status::internal(
                                    format!("Batch: {}", e)
                                )));
                                return;
                            }
                            if let Err(e) = ctx.decode(&mut batch) {
                                let _ = tx.blocking_send(Err(Status::internal(
                                    format!("Decode prompt: {}", e)
                                )));
                                return;
                            }

                            // total_cmp gives a total ordering — partial_cmp().unwrap()
                            // panics if any logit is NaN.
                            let argmax = |logits: &[f32]| -> i32 {
                                logits.iter()
                                    .enumerate()
                                    .max_by(|(_, a), (_, b)| a.total_cmp(b))
                                    .map(|(i, _)| i as i32)
                                    .unwrap_or(0)
                            };

                            // the model's real end-of-generation token (was hardcoded 2)
                            let eos_id = model.token_eos().0;
                            let mut last_token = llama_cpp_4::token::LlamaToken(0);
                            for pos in 0..max_tokens {
                                if last_token.0 != 0 {
                                    let mut next_batch = LlamaBatch::new(1, 1);
                                    if next_batch.add(last_token, pos as i32, &[0], true).is_err() {
                                        break;
                                    }
                                    if ctx.decode(&mut next_batch).is_err() {
                                        break;
                                    }
                                }

                                let logits = ctx.get_logits();
                                if logits.is_empty() {
                                    break;
                                }

                                let sampled_id = argmax(logits);
                                last_token = llama_cpp_4::token::LlamaToken(sampled_id);

                                let text = model.token_to_str(last_token, Special::Tokenize)
                                    .unwrap_or_else(|_| format!("[{}]", sampled_id));

                                if tx.blocking_send(Ok(ForwardResponse {
                                    hidden_state: vec![],
                                    logits: text.as_bytes().to_vec(),
                                    has_logits: true,
                                    compute_time_ms: 0,
                                })).is_err() {
                                    break;
                                }

                                if sampled_id == eos_id { break; }
                            }

                            let _ = tx.blocking_send(Ok(ForwardResponse {
                                hidden_state: vec![],
                                logits: vec![],
                                has_logits: false,
                                compute_time_ms: 0,
                            }));
                        });

                        let mut rx_stream = ReceiverStream::new(rx);
                        while let Some(resp) = rx_stream.next().await {
                            yield resp?;
                        }
                        continue;
                    }

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

    async fn get_rpc_endpoint(
        &self,
        _request: Request<Empty>,
    ) -> Result<Response<RpcEndpoint>, Status> {
        let running = self.rpc_port > 0;
        Ok(Response::new(RpcEndpoint {
            running,
            port: self.rpc_port as u32,
            // Address is filled in by the orchestrator which knows the node's IP.
            address: String::new(),
        }))
    }
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
