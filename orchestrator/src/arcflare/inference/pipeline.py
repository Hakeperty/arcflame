import asyncio
import logging
import os
from typing import AsyncGenerator, Optional

logger = logging.getLogger("arcflare.inference")


class InferencePipeline:
    """Coordinates the distributed inference pipeline across nodes.

    Pipeline flow:
    1. Tokenize input on orchestrator
    2. Send to first node (embedding + early layers)
    3. Forward hidden states through intermediate nodes
    4. Final node runs LM head and returns logits
    5. Orchestrator decodes and streams response
    """

    def __init__(self):
        self.active_pipelines: dict = {}
        self.node_connections: dict = {}

    def _find_model(self, model_name: str) -> Optional[str]:
        models_dir = os.environ.get(
            "ARCFLARE_MODELS_DIR",
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "models"),
        )
        models_dir = os.path.abspath(models_dir)

        # Map model name to GGUF file
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

    async def run(
        self,
        model: str,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> str:
        """Run inference and return full result."""
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
        """Stream tokens from inference pipeline."""
        from ..main import discovery_service

        nodes = discovery_service.get_nodes() if discovery_service else []
        if not nodes:
            logger.info("No cluster nodes available — using local inference")
            async for token in self._local_inference(prompt, max_tokens, temperature):
                yield token
            return

        logger.info(f"Running inference on {len(nodes)} nodes (Phase 2 — using local fallback)")
        async for token in self._local_inference(prompt, max_tokens, temperature):
            yield token

    async def _local_inference(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> AsyncGenerator[str, None]:
        """Run inference using local llama-cli subprocess."""
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
            # Extract the model response: take everything between "> prompt" and "[Prompt:" status
            lines = output.split("\n")
            capturing = False
            for line in lines:
                stripped = line.strip()
                if not stripped:
                    continue
                # Start capturing when we see the "> prompt" line
                if not capturing and stripped.startswith("> "):
                    capturing = True
                    continue
                # Stop at status line
                if capturing and stripped.startswith("["):
                    break
                # Skip help text between prompt and response
                if capturing and stripped.startswith("/"):
                    continue
                # Capture response lines, stripping spinner artifacts
                if capturing and stripped:
                    # Strip spinner character + backspace artifacts
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

    async def _distribute_prompt(self, prompt: str, nodes: list) -> list:
        """Tokenize and distribute prompt across nodes."""
        return []

    async def _collect_logits(self, nodes: list) -> list:
        """Collect logits from the final node in pipeline."""
        return []


# Global pipeline instance
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
