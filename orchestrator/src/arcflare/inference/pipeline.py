import asyncio
import logging
import os
import random
from typing import AsyncGenerator, Optional

from .grpc_client import NodeGrpcClient

logger = logging.getLogger("arcflare.inference")


class InferencePipeline:
    """Coordinates distributed inference across the cluster.

    Pipeline modes:
    1. Local fallback — runs llama-cli subprocess on orchestrator
    2. Distributed — load-balances requests across available node agents
    3. (Future) True pipeline parallelism via per-layer shards
    """

    def __init__(self):
        self.active_pipelines: dict = {}
        self.node_connections: dict[str, NodeGrpcClient] = {}

    def _find_model(self, model_name: str) -> Optional[str]:
        models_dir = os.environ.get(
            "ARCFLARE_MODELS_DIR",
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "models"),
        )
        models_dir = os.path.abspath(models_dir)

        for fname in os.listdir(models_dir):
            if fname.endswith(".gguf"):
                return os.path.join(models_dir, fname)

        return None

    def _find_llama_cli(self) -> Optional[str]:
        candidates = [
            os.environ.get("ARCFLARE_LLAMA_CLI", ""),
            "/usr/local/bin/llama-cli",
            "/usr/local/bin/llama",
            "/tmp/llama-cli",
            "/app/llama-cli",
            "/app/llama",
        ]
        for path in candidates:
            if path and os.path.isfile(path) and os.access(path, os.X_OK):
                return path
        return None

    def _get_models_dir(self) -> str:
        return os.environ.get(
            "ARCFLARE_MODELS_DIR",
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "models"),
        )

    async def _connect_to_nodes(self, nodes: list[dict]):
        """Establish gRPC connections to available nodes."""
        for node in nodes:
            node_id = node.get("node_id", "")
            if node_id and node_id not in self.node_connections:
                ip = node.get("ip_address", "")
                port = node.get("grpc_port", 9001)
                if ip and port:
                    client = NodeGrpcClient(node_id, ip, port)
                    ok = await client.connect()
                    if ok:
                        self.node_connections[node_id] = client
                        logger.info(f"gRPC connected to node {node_id}")

    async def run(
        self,
        model: str,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> str:
        tokens = []
        async for token in self.run_stream(model, prompt, max_tokens, temperature):
            tokens.append(token)
        return "".join(tokens)

    async def run_stream(
        self,
        model: str,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> AsyncGenerator[str, None]:
        from ..main import discovery_service

        nodes = discovery_service.get_nodes() if discovery_service else []
        alive = [n for n in nodes if n.get("grpc_port") and n.get("ip_address")]
        active_connections = list(self.node_connections.keys())

        # Try distributed inference if we have connected nodes with shards loaded
        if alive and active_connections:
            logger.info(f"Distributed mode: {len(alive)} nodes available")
            try:
                await self._connect_to_nodes(alive)
                async for token in self._distributed_inference(
                    model, prompt, max_tokens, temperature, alive
                ):
                    yield token
                return
            except Exception as e:
                logger.warning(f"Distributed inference failed ({e}), falling back to local")

        logger.info("Local inference mode")
        async for token in self._local_inference(prompt, max_tokens, temperature):
            yield token

    async def _load_shards_on_nodes(
        self, model_path: str, nodes: list[dict]
    ) -> bool:
        """Attempt to load the model on each connected node via gRPC.
        Returns True if at least one node loaded successfully (for validation)."""
        loaded = 0
        for node in nodes:
            node_id = node.get("node_id", "")
            client = self.node_connections.get(node_id)
            if not client:
                continue
            total_layers = self._get_model_layer_count(model_path) or 24
            n_nodes = max(1, len(nodes))
            layers_per = total_layers // n_nodes
            first = nodes.index(node) * layers_per
            num = layers_per if nodes.index(node) < n_nodes - 1 else total_layers - first
            has_head = nodes.index(node) == n_nodes - 1

            status = await client.load_shard(
                model_name="arcflare/default",
                gguf_path=model_path,
                first_layer=first,
                num_layers=num,
                has_lm_head=has_head,
            )
            if status and status.loaded:
                loaded += 1
        return loaded > 0

    def _get_model_layer_count(self, model_path: str) -> Optional[int]:
        """Quick heuristic: try gguf-splitter, or guess from model size."""
        try:
            import subprocess
            result = subprocess.run(
                ["gguf-splitter", "--model", model_path],
                capture_output=True, text=True, timeout=10,
            )
            for line in result.stdout.split("\n"):
                if line.startswith("Total layers:"):
                    return int(line.split(":")[1].strip())
        except Exception:
            pass
        # Fallback: common layer counts
        file_size = os.path.getsize(model_path)
        if file_size < 1_000_000_000:
            return 24  # ~0.5B params: 24 layers
        return 32

    async def _distributed_inference(
        self,
        model: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
        nodes: list[dict],
    ) -> AsyncGenerator[str, None]:
        """Run inference distributed across nodes.
        Falls back to local llama-cli if distribution fails."""
        model_path = self._find_model(model)
        if not model_path:
            logger.warning("No model found for distributed inference")
            return

        # Try to load shards on nodes
        await self._connect_to_nodes(nodes)
        shards_loaded = await self._load_shards_on_nodes(model_path, nodes)

        if shards_loaded:
            logger.info("Shards loaded on nodes, trying gRPC inference")
            async for token in self._grpc_inference(prompt, max_tokens, temperature, nodes):
                yield token
            return

        # Fall back to local if no nodes accepted shards
        logger.info("gRPC shard loading failed, using local llama-cli")
        async for token in self._local_inference(prompt, max_tokens, temperature):
            yield token

    async def _grpc_inference(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        nodes: list[dict],
    ) -> AsyncGenerator[str, None]:
        """Send prompt through node pipeline via gRPC Forward calls."""
        first_node = nodes[0] if nodes else None
        if not first_node:
            return

        client = self.node_connections.get(first_node.get("node_id", ""))
        if not client:
            return

        # Send embedding request to first node
        resp = await client.forward(
            hidden_state=prompt.encode("utf-8"),
            start_layer=0,
            num_layers=1,
            input_ids=[1],
        )

        if resp and resp.logits:
            # Try to decode logits into text
            try:
                text = resp.logits.decode("utf-8", errors="replace")
                for char in text:
                    yield char
                    await asyncio.sleep(0.01)
                return
            except Exception:
                pass

        # Fallback: local inference
        logger.info("gRPC inference returned no results, using local fallback")
        async for token in self._local_inference(prompt, max_tokens, temperature):
            yield token

    async def _local_inference(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> AsyncGenerator[str, None]:
        model_path = self._find_model("default")
        llama_cli = self._find_llama_cli()

        if not model_path or not llama_cli:
            logger.warning(f"Model ({model_path}) or llama-cli ({llama_cli}) not found — using stub")
            response = (
                f"[ArcFlare local mode]\n"
                f"Received prompt ({len(prompt)} chars). "
                f"Install llama-cli and a GGUF model for real inference.\n"
            )
            for char in response:
                yield char
                await asyncio.sleep(0.01)
            return

        logger.info(f"Running llama-cli with model: {model_path}")
        cmd = [
            llama_cli,
            "-m", model_path,
            "-p", prompt,
            "-n", str(max_tokens),
            "--no-display-prompt",
            "--single-turn",
        ]

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout_data, _ = await asyncio.wait_for(
                proc.communicate(), timeout=600
            )

            output = stdout_data.decode("utf-8", errors="replace")
            lines = output.split("\n")
            capturing = False
            for line in lines:
                stripped = line.strip()
                if not stripped:
                    continue
                if not capturing and stripped.startswith("> "):
                    capturing = True
                    continue
                if capturing and stripped.startswith("["):
                    break
                if capturing and stripped.startswith("/"):
                    continue
                if capturing and stripped:
                    if any(stripped.startswith(c) for c in ("|", "/", "-", "\\", "=")):
                        stripped = stripped.lstrip("|/-\\=").lstrip("\b \b").strip()
                    clean = stripped.replace("\b", "").replace("\r", "").strip()
                    if clean and not clean.startswith(">") and not clean.startswith("Exiting"):
                        yield clean + "\n"
                        await asyncio.sleep(0.01)

        except asyncio.TimeoutError:
            logger.error("llama-cli timed out")
            yield "\n[Inference timed out]\n"
        except FileNotFoundError:
            logger.error("llama-cli binary not found")
            yield "\n[llama-cli not found]\n"
        except Exception as e:
            logger.error(f"Inference error: {e}")
            yield f"\n[Error: {e}]\n"


_pipeline: Optional[InferencePipeline] = None


def get_pipeline() -> InferencePipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = InferencePipeline()
    return _pipeline


async def run_inference(
    model: str,
    prompt: str,
    max_tokens: int = 1024,
    temperature: float = 0.7,
) -> str:
    pipeline = get_pipeline()
    return await pipeline.run(model, prompt, max_tokens, temperature)


async def run_inference_stream(
    model: str,
    prompt: str,
    max_tokens: int = 1024,
    temperature: float = 0.7,
) -> AsyncGenerator[str, None]:
    pipeline = get_pipeline()
    async for token in pipeline.run_stream(model, prompt, max_tokens, temperature):
        yield token
