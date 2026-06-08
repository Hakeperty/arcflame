import asyncio
import logging
import os
import random
import shutil
from typing import AsyncGenerator, Optional

from .grpc_client import NodeGrpcClient

logger = logging.getLogger("arcflare.inference")


class InferencePipeline:
    """Coordinates distributed inference across the cluster.

    Pipeline modes (tried in order):
    1. RPC distributed — orchestrator runs llama-cli with --rpc <node1>,<node2>,...
       Each node must run llama-rpc-server (--enable-rpc on the node agent).
       Requires 1+ nodes with rpc_port set. llama-cli keeps all computation on
       the nodes; the model is split by tensor automatically.
    2. gRPC streaming — sends generation to one node via our ForwardStream proto.
    3. Local fallback — runs llama-cli subprocess on the orchestrator itself.
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
        rpc_endpoints = discovery_service.get_rpc_endpoints() if discovery_service else []

        # Mode 1: RPC distributed — requires at least one rpc-server endpoint
        if rpc_endpoints:
            logger.info(f"RPC distributed mode: {len(rpc_endpoints)} endpoint(s): {rpc_endpoints}")
            try:
                async for token in self._rpc_distributed_inference(
                    model, prompt, max_tokens, temperature, rpc_endpoints
                ):
                    yield token
                return
            except Exception as e:
                logger.warning(f"RPC distributed inference failed ({e}), falling back")

        # Mode 2: gRPC streaming to a single node
        alive = [n for n in nodes if n.get("grpc_port") and n.get("ip_address")]
        active_connections = list(self.node_connections.keys())
        if alive and active_connections:
            logger.info(f"gRPC stream mode: {len(alive)} nodes available")
            try:
                await self._connect_to_nodes(alive)
                async for token in self._distributed_inference(
                    model, prompt, max_tokens, temperature, alive
                ):
                    yield token
                return
            except Exception as e:
                logger.warning(f"gRPC inference failed ({e}), falling back to local")

        # Mode 3: local llama-cli
        logger.info("Local inference mode")
        async for token in self._local_inference(prompt, max_tokens, temperature):
            yield token

    # ─── RPC distributed inference ───

    async def _rpc_distributed_inference(
        self,
        model: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
        rpc_endpoints: list[str],
    ) -> AsyncGenerator[str, None]:
        """Run llama-cli with --rpc pointing at all node rpc-servers.

        llama-cli offloads tensor shards to each rpc-server automatically,
        distributing model layers across the cluster.
        """
        model_path = self._find_model(model)
        llama_cli = self._find_llama_cli()

        if not model_path:
            raise RuntimeError("No model found for RPC inference")
        if not llama_cli:
            raise RuntimeError("llama-cli binary not found for RPC inference")

        rpc_arg = ",".join(rpc_endpoints)
        logger.info(f"Invoking llama-cli --rpc {rpc_arg} with model {model_path}")

        cmd = [
            llama_cli,
            "-m", model_path,
            "--rpc", rpc_arg,
            "-p", prompt,
            "-n", str(max_tokens),
            "--no-display-prompt",
            "--single-turn",
        ]

        if temperature > 0:
            cmd += ["--temp", str(temperature)]
        else:
            cmd += ["--greedy"]

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout_data, stderr_data = await asyncio.wait_for(
                proc.communicate(), timeout=600
            )

            if proc.returncode != 0:
                stderr_text = stderr_data.decode("utf-8", errors="replace")
                raise RuntimeError(f"llama-cli exited {proc.returncode}: {stderr_text[:200]}")

            output = stdout_data.decode("utf-8", errors="replace")
            capturing = False
            for line in output.split("\n"):
                stripped = line.strip()
                if not stripped:
                    continue
                if not capturing and stripped.startswith("> "):
                    capturing = True
                    continue
                if capturing and stripped.startswith("["):
                    break
                if capturing and stripped.startswith("Exiting"):
                    break
                if capturing and stripped:
                    clean = stripped.lstrip("|/-\\").replace("\b", "").replace("\r", "").strip()
                    if clean:
                        yield clean + "\n"
                        await asyncio.sleep(0.005)

        except asyncio.TimeoutError:
            logger.error("llama-cli RPC inference timed out")
            raise RuntimeError("RPC inference timed out")

    # ─── gRPC pipeline (single-node fallback) ───

    async def _load_shards_on_nodes(
        self, model_path: str, nodes: list[dict]
    ) -> bool:
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
        file_size = os.path.getsize(model_path)
        if file_size < 1_000_000_000:
            return 24
        return 32

    async def _distributed_inference(
        self,
        model: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
        nodes: list[dict],
    ) -> AsyncGenerator[str, None]:
        model_path = self._find_model(model)
        if not model_path:
            logger.warning("No model found for distributed inference")
            return

        await self._connect_to_nodes(nodes)
        shards_loaded = await self._load_shards_on_nodes(model_path, nodes)

        if shards_loaded:
            logger.info("Shards loaded on nodes, trying gRPC inference")
            async for token in self._grpc_inference(prompt, max_tokens, temperature, nodes):
                yield token
            return

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
        target = self._pick_target_node(nodes)
        if not target:
            logger.warning("No available node for inference")
            return

        node_id = target.get("node_id", "")
        client = self.node_connections.get(node_id)
        if not client:
            logger.warning(f"Node {node_id} not connected")
            return

        logger.info(f"Streaming inference to node {node_id}")

        try:
            async for chunk in client.forward_stream(
                text_prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            ):
                if chunk.logits:
                    text = chunk.logits.decode("utf-8", errors="replace")
                    for char in text:
                        yield char
                        await asyncio.sleep(0.005)
                elif not chunk.has_logits:
                    break
        except Exception as e:
            logger.warning(f"gRPC streaming inference failed: {e}")
            logger.info("Falling back to local inference")
            async for token in self._local_inference(prompt, max_tokens, temperature):
                yield token

    def _pick_target_node(self, nodes: list[dict]) -> Optional[dict]:
        connected = [
            n for n in nodes
            if n.get("node_id", "") in self.node_connections
        ]
        if not connected:
            return None
        return random.choice(connected)

    # ─── Local fallback ───

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
