"""Persistent llama-server manager (optimization for finding #27/#30).

Instead of spawning a fresh `llama-cli` per request — which reloads the entire
GGUF from disk and re-ships tensors to the RPC nodes every time (~tens of
seconds) — we keep a single `llama-server` process alive. The model loads once;
subsequent requests stream tokens from its HTTP API with no reload.

The server is (re)started only when the (model, rpc_endpoints) configuration
changes. Distributed inference is enabled by passing `--rpc <ep1>,<ep2>` so the
same persistent server splits tensors across the cluster.
"""
import asyncio
import json
import logging
import os
from typing import AsyncGenerator, Optional

import httpx

logger = logging.getLogger("arcflare.inference.server")

_HOST = "127.0.0.1"
_DEFAULT_PORT = int(os.environ.get("ARCFLARE_LLAMA_SERVER_PORT", "8081"))


def find_llama_server() -> Optional[str]:
    """Locate the llama-server binary."""
    candidates = [
        os.environ.get("ARCFLARE_LLAMA_SERVER", ""),
    ]
    # sibling of llama-cli, if that env is set
    cli = os.environ.get("ARCFLARE_LLAMA_CLI", "")
    if cli:
        candidates.append(os.path.join(os.path.dirname(cli), "llama-server"))
    candidates += [
        "/usr/local/bin/llama-server",
        "/opt/llama/bin/llama-server",
        "/app/llama-server",
    ]
    for path in candidates:
        if path and os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    return None


class LlamaServerManager:
    def __init__(self, port: int = _DEFAULT_PORT):
        self.port = port
        self.base_url = f"http://{_HOST}:{port}"
        self._proc: Optional[asyncio.subprocess.Process] = None
        self._config: Optional[tuple] = None  # (model_path, tuple(sorted(rpc_endpoints)))
        self._lock = asyncio.Lock()

    @property
    def available(self) -> bool:
        return find_llama_server() is not None

    async def ensure(self, model_path: str, rpc_endpoints: list[str], ctx_size: int = 2048) -> bool:
        """Ensure a llama-server is running for this config. Returns True if ready."""
        want = (model_path, tuple(sorted(rpc_endpoints)))
        async with self._lock:
            if self._proc is not None and self._proc.returncode is None and self._config == want:
                return True  # already running with the right config
            # config changed or process dead → (re)start
            await self._stop_locked()
            return await self._start_locked(model_path, list(rpc_endpoints), ctx_size, want)

    async def _start_locked(self, model_path, rpc_endpoints, ctx_size, want) -> bool:
        binary = find_llama_server()
        if not binary:
            return False
        cmd = [
            binary, "-m", model_path,
            "--host", _HOST, "--port", str(self.port),
            "-c", str(ctx_size),
        ]
        if rpc_endpoints:
            cmd += ["--rpc", ",".join(rpc_endpoints)]

        # make sure the bundled .so libs resolve
        env = dict(os.environ)
        libdir = os.path.dirname(binary)
        env["LD_LIBRARY_PATH"] = libdir + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")

        logger.info(f"Starting llama-server: {' '.join(cmd)}")
        self._proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL, env=env,
        )
        self._config = want
        ready = await self._wait_healthy(timeout=240)
        if not ready:
            logger.warning("llama-server did not become healthy in time")
            await self._stop_locked()
            return False
        logger.info(f"llama-server ready at {self.base_url}")
        return True

    async def _wait_healthy(self, timeout: float) -> bool:
        deadline = timeout
        step = 1.0
        async with httpx.AsyncClient(timeout=5.0) as client:
            elapsed = 0.0
            while elapsed < deadline:
                if self._proc is None or self._proc.returncode is not None:
                    return False  # process died during load
                try:
                    r = await client.get(f"{self.base_url}/health")
                    if r.status_code == 200:
                        return True
                except (httpx.HTTPError, OSError):
                    pass
                await asyncio.sleep(step)
                elapsed += step
        return False

    async def stream_completion(
        self, prompt: str, n_predict: int, temperature: float,
    ) -> AsyncGenerator[str, None]:
        """Stream tokens from the running server's /completion endpoint."""
        payload = {
            "prompt": prompt,
            "n_predict": n_predict,
            "temperature": temperature,
            "stream": True,
            "cache_prompt": True,
            # stop the model from hallucinating further turns of the dialogue
            "stop": ["User:", "USER:", "user:", "<|user|>", "System:", "\nAssistant:", "\nQ:"],
        }
        async with httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=10.0)) as client:
            async with client.stream("POST", f"{self.base_url}/completion", json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[len("data:"):].strip()
                    if not data:
                        continue
                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    content = obj.get("content", "")
                    if content:
                        yield content
                    if obj.get("stop"):
                        break

    async def stop(self):
        async with self._lock:
            await self._stop_locked()

    async def _stop_locked(self):
        if self._proc is not None and self._proc.returncode is None:
            try:
                self._proc.terminate()
                try:
                    await asyncio.wait_for(self._proc.wait(), timeout=10)
                except asyncio.TimeoutError:
                    self._proc.kill()
                    await self._proc.wait()
            except ProcessLookupError:
                pass
        self._proc = None
        self._config = None
