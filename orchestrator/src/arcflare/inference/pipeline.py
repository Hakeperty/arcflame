import asyncio
import logging
import os
import random
from functools import lru_cache
from typing import AsyncGenerator, Optional

from .grpc_client import NodeGrpcClient
from .llama_server import LlamaServerManager

logger = logging.getLogger("arcflare.inference")


def _models_dir() -> str:
    return os.path.abspath(os.environ.get(
        "ARCFLARE_MODELS_DIR",
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "models"),
    ))


@lru_cache(maxsize=1)
def _find_llama_cli_cached() -> Optional[str]:
    candidates = [
        os.environ.get("ARCFLARE_LLAMA_CLI", ""),
        "/usr/local/bin/llama-cli",
        "/opt/llama/bin/llama-cli",
        "/usr/local/bin/llama",
        "/app/llama-cli",
    ]
    for path in candidates:
        if path and os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    return None


class InferencePipeline:
    """Coordinates distributed inference across the cluster.

    Pipeline modes (tried in order):
    1. llama-server (persistent) — a single long-lived server loads the model
       once and streams tokens over HTTP. `--rpc <ep1>,<ep2>` makes it split
       tensors across the cluster (distributed); with no endpoints it runs
       locally. This avoids reloading the model on every request.
    2. llama-cli subprocess — fallback that runs `llama-cli [--rpc ...]` once per
       request (reloads the model each time). Used only if llama-server is absent.
    3. gRPC ForwardStream — custom per-node streaming protocol.
    4. Stub — no binary/model available.
    """

    def __init__(self):
        self.node_connections: dict[str, NodeGrpcClient] = {}
        self.server = LlamaServerManager()

    # ─── lookups (cached) ───

    def _find_model(self, model_name: str) -> Optional[str]:
        """Resolve a model name to a .gguf path. Uses the name when it matches a
        file; otherwise falls back to the first model (sorted, deterministic)."""
        models_dir = _models_dir()
        if not os.path.isdir(models_dir):
            return None
        ggufs = sorted(f for f in os.listdir(models_dir) if f.endswith(".gguf"))
        if not ggufs:
            return None
        wanted = (model_name or "").split("/")[-1].removesuffix(".gguf").lower()
        if wanted and wanted != "default":
            for f in ggufs:
                if wanted in f.lower():
                    return os.path.join(models_dir, f)
        return os.path.join(models_dir, ggufs[0])

    def _find_llama_cli(self) -> Optional[str]:
        return _find_llama_cli_cached()

    async def _connect_to_nodes(self, nodes: list[dict]):
        for node in nodes:
            node_id = node.get("node_id", "")
            if node_id and node_id not in self.node_connections:
                ip = node.get("ip_address", "")
                port = node.get("grpc_port", 9001)
                if ip and port:
                    client = NodeGrpcClient(node_id, ip, port)
                    if await client.connect():
                        self.node_connections[node_id] = client
                        logger.info(f"gRPC connected to node {node_id}")

    # ─── public API ───

    async def run(self, model: str, prompt: str, max_tokens: int = 1024,
                  temperature: float = 0.7) -> str:
        tokens = []
        async for token in self.run_stream(model, prompt, max_tokens, temperature):
            tokens.append(token)
        return "".join(tokens)

    async def run_stream(self, model: str, prompt: str, max_tokens: int = 1024,
                         temperature: float = 0.7) -> AsyncGenerator[str, None]:
        from ..main import discovery_service

        nodes = discovery_service.get_nodes() if discovery_service else []
        rpc_endpoints = discovery_service.get_rpc_endpoints() if discovery_service else []
        model_path = self._find_model(model)

        # Mode 1: persistent llama-server (handles both distributed and local)
        if model_path and self.server.available:
            mode = f"rpc_distributed ({len(rpc_endpoints)} nodes)" if rpc_endpoints else "local"
            logger.info(f"llama-server mode [{mode}], model={model_path}")
            try:
                if await self.server.ensure(model_path, rpc_endpoints):
                    async for tok in self.server.stream_completion(prompt, max_tokens, temperature):
                        yield tok
                    return
                logger.warning("llama-server not ready, falling back")
            except Exception as e:
                logger.warning(f"llama-server inference failed ({e}), falling back")

        # Mode 2: one-shot llama-cli (with --rpc if endpoints exist)
        if model_path and self._find_llama_cli():
            try:
                async for tok in self._llama_cli_inference(
                    model_path, prompt, max_tokens, temperature, rpc_endpoints):
                    yield tok
                return
            except Exception as e:
                logger.warning(f"llama-cli inference failed ({e}), falling back")

        # Mode 3: gRPC streaming to a connected node
        alive = [n for n in nodes if n.get("grpc_port") and n.get("ip_address")]
        if alive:
            await self._connect_to_nodes(alive)
            if self.node_connections:
                logger.info(f"gRPC stream mode: {len(alive)} nodes")
                try:
                    async for tok in self._distributed_inference(
                        model, prompt, max_tokens, temperature, alive):
                        yield tok
                    return
                except Exception as e:
                    logger.warning(f"gRPC inference failed ({e}), falling back")

        # Mode 4: stub
        logger.info("No inference backend available — stub response")
        async for tok in self._stub(prompt):
            yield tok

    # ─── llama-cli fallback (one-shot) ───

    async def _llama_cli_inference(
        self, model_path: str, prompt: str, max_tokens: int, temperature: float,
        rpc_endpoints: list[str],
    ) -> AsyncGenerator[str, None]:
        llama_cli = self._find_llama_cli()
        cmd = [llama_cli, "-m", model_path, "-p", prompt, "-n", str(max_tokens),
               "--no-display-prompt", "--single-turn"]
        if rpc_endpoints:
            cmd += ["--rpc", ",".join(rpc_endpoints)]
        if temperature > 0:
            cmd += ["--temp", str(temperature)]

        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        try:
            stdout_data, stderr_data = await asyncio.wait_for(proc.communicate(), timeout=600)
            if proc.returncode != 0:
                err = stderr_data.decode("utf-8", errors="replace")[:200]
                raise RuntimeError(f"llama-cli exited {proc.returncode}: {err}")
            for line in self._strip_banner(stdout_data.decode("utf-8", errors="replace")):
                yield line
                await asyncio.sleep(0.005)
        except asyncio.TimeoutError:
            logger.error("llama-cli timed out")
            raise RuntimeError("inference timed out")
        finally:
            # always reap the child — on timeout, early-return, or consumer disconnect
            if proc.returncode is None:
                try:
                    proc.kill()
                    await proc.wait()
                except ProcessLookupError:
                    pass

    @staticmethod
    def _strip_banner(output: str):
        """Extract model output from llama-cli's noisy stdout (banner/spinner/footer)."""
        capturing = False
        for line in output.split("\n"):
            stripped = line.strip()
            if not stripped:
                continue
            if not capturing and stripped.startswith("> "):
                capturing = True
                continue
            if capturing and (stripped.startswith("[") or stripped.startswith("Exiting")):
                break
            if capturing:
                clean = stripped.lstrip("|/-\\").replace("\b", "").replace("\r", "").strip()
                if clean:
                    yield clean + "\n"

    # ─── gRPC pipeline ───

    async def _load_shards_on_nodes(self, model_path: str, nodes: list[dict]) -> bool:
        loaded = 0
        total_layers = self._get_model_layer_count(model_path) or 24
        n_nodes = max(1, len(nodes))
        layers_per = total_layers // n_nodes
        for idx, node in enumerate(nodes):
            client = self.node_connections.get(node.get("node_id", ""))
            if not client:
                continue
            first = idx * layers_per
            is_last = idx == n_nodes - 1
            num = total_layers - first if is_last else layers_per
            status = await client.load_shard(
                model_name="arcflare/default", gguf_path=model_path,
                first_layer=first, num_layers=num, has_lm_head=is_last)
            if status and status.loaded:
                loaded += 1
        return loaded > 0

    @lru_cache(maxsize=8)
    def _get_model_layer_count(self, model_path: str) -> Optional[int]:
        try:
            import subprocess
            result = subprocess.run(["gguf-splitter", "--model", model_path],
                                    capture_output=True, text=True, timeout=10)
            for line in result.stdout.split("\n"):
                if line.startswith("Total layers:"):
                    return int(line.split(":")[1].strip())
        except Exception:
            pass
        return 24 if os.path.getsize(model_path) < 1_000_000_000 else 32

    async def _distributed_inference(self, model, prompt, max_tokens, temperature, nodes):
        model_path = self._find_model(model)
        if not model_path:
            logger.warning("No model for gRPC inference, using stub")
            async for tok in self._stub(prompt):
                yield tok
            return
        await self._connect_to_nodes(nodes)
        if await self._load_shards_on_nodes(model_path, nodes):
            async for tok in self._grpc_inference(prompt, max_tokens, temperature, nodes):
                yield tok
            return
        logger.info("gRPC shard load failed, using stub")
        async for tok in self._stub(prompt):
            yield tok

    async def _grpc_inference(self, prompt, max_tokens, temperature, nodes):
        target = self._pick_target_node(nodes)
        client = self.node_connections.get(target.get("node_id", "")) if target else None
        if not client:
            logger.warning("No connected node for gRPC inference")
            async for tok in self._stub(prompt):
                yield tok
            return
        try:
            async for chunk in client.forward_stream(
                text_prompt=prompt, max_tokens=max_tokens, temperature=temperature):
                if chunk.logits:
                    yield chunk.logits.decode("utf-8", errors="replace")
                elif not chunk.has_logits:
                    break
        except Exception as e:
            logger.warning(f"gRPC streaming failed: {e}")
            raise

    def _pick_target_node(self, nodes: list[dict]) -> Optional[dict]:
        connected = [n for n in nodes if n.get("node_id", "") in self.node_connections]
        return random.choice(connected) if connected else None

    # ─── stub ───

    async def _stub(self, prompt: str) -> AsyncGenerator[str, None]:
        msg = (f"[ArcFlare] No inference backend available for a {len(prompt)}-char prompt. "
               f"Install llama-server/llama-cli and provide a GGUF model.\n")
        for chunk in msg.split(" "):
            yield chunk + " "
            await asyncio.sleep(0.01)


_pipeline: Optional[InferencePipeline] = None


def get_pipeline() -> InferencePipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = InferencePipeline()
    return _pipeline


async def run_inference(model: str, prompt: str, max_tokens: int = 1024,
                        temperature: float = 0.7) -> str:
    return await get_pipeline().run(model, prompt, max_tokens, temperature)


async def run_inference_stream(model: str, prompt: str, max_tokens: int = 1024,
                               temperature: float = 0.7) -> AsyncGenerator[str, None]:
    async for token in get_pipeline().run_stream(model, prompt, max_tokens, temperature):
        yield token
