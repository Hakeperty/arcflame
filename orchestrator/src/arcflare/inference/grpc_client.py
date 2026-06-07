"""gRPC client for communicating with ArcFlare node agents."""

import logging
from typing import Optional

import grpc

from ..arcflare_pb2 import (
    Empty,
    ForwardRequest,
    ForwardResponse,
    LoadStatus,
    ShardConfig,
)
from ..arcflare_pb2_grpc import NodeAgentStub

logger = logging.getLogger("arcflare.grpc_client")


class NodeGrpcClient:
    """gRPC connection to a single node agent."""

    def __init__(self, node_id: str, host: str, grpc_port: int):
        self.node_id = node_id
        self.address = f"{host}:{grpc_port}"
        self._channel: Optional[grpc.aio.Channel] = None
        self._stub: Optional[NodeAgentStub] = None

    async def connect(self) -> bool:
        try:
            self._channel = grpc.aio.insecure_channel(self.address)
            self._stub = NodeAgentStub(self._channel)
            # Verify connection with a lightweight call
            await self._stub.GetHardwareInfo(Empty(), timeout=5)
            logger.info(f"Connected to node {self.node_id} at {self.address}")
            return True
        except Exception as e:
            logger.warning(f"Failed to connect to node {self.node_id} at {self.address}: {e}")
            self._channel = None
            self._stub = None
            return False

    async def load_shard(
        self,
        model_name: str,
        gguf_path: str,
        first_layer: int,
        num_layers: int,
        has_lm_head: bool,
        max_context: int = 4096,
    ) -> Optional[LoadStatus]:
        if not self._stub:
            return None
        try:
            config = ShardConfig(
                model_name=model_name,
                shard_path="",
                first_layer=first_layer,
                num_layers=num_layers,
                has_lm_head=has_lm_head,
                max_context_length=max_context,
                gguf_path=gguf_path,
                peer_addresses=[],
            )
            status = await self._stub.LoadShard(config, timeout=120)
            return status
        except Exception as e:
            logger.error(f"LoadShard on {self.node_id} failed: {e}")
            return None

    async def forward(
        self,
        hidden_state: bytes,
        start_layer: int,
        num_layers: int,
        input_ids: Optional[list[int]] = None,
    ) -> Optional[ForwardResponse]:
        if not self._stub:
            return None
        try:
            req = ForwardRequest(
                hidden_state=hidden_state,
                start_layer=start_layer,
                num_layers=num_layers,
                input_ids=input_ids or [],
            )
            resp = await self._stub.Forward(req, timeout=60)
            return resp
        except Exception as e:
            logger.error(f"Forward on {self.node_id} failed: {e}")
            return None

    async def unload_shard(self) -> bool:
        if not self._stub:
            return False
        try:
            await self._stub.UnloadShard(Empty(), timeout=10)
            return True
        except Exception as e:
            logger.error(f"UnloadShard on {self.node_id} failed: {e}")
            return False

    async def get_inference_stats(self) -> Optional[dict]:
        if not self._stub:
            return None
        try:
            stats = await self._stub.GetInferenceStats(Empty(), timeout=5)
            return {
                "layers_loaded": stats.layers_loaded,
                "total_forward_calls": stats.total_forward_calls,
                "avg_forward_time_ms": stats.avg_forward_time_ms,
                "total_tokens_processed": stats.total_tokens_processed,
                "kv_cache_used_bytes": stats.kv_cache_used_bytes,
                "peak_memory_bytes": stats.peak_memory_bytes,
            }
        except Exception as e:
            logger.error(f"GetInferenceStats on {self.node_id} failed: {e}")
            return None

    async def close(self):
        if self._channel:
            await self._channel.close()
            self._channel = None
            self._stub = None
