import asyncio
import logging
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
            logger.info("No cluster nodes available — using local inference stub")
            async for token in self._local_inference_stub(prompt, max_tokens):
                yield token
            return

        # TODO: Phase 2 — actual distributed pipeline
        # 1. Get partition plan
        # 2. Load shards on nodes
        # 3. Send prompt to first node
        # 4. Stream results back through chain

        logger.info(f"Running inference on {len(nodes)} nodes (stub)")
        async for token in self._local_inference_stub(prompt, max_tokens):
            yield token

    async def _local_inference_stub(
        self,
        prompt: str,
        max_tokens: int = 1024,
    ) -> AsyncGenerator[str, None]:
        """Stub for when no nodes are available."""
        response = (
            f"[ArcFlare local mode]\n"
            f"Received prompt ({len(prompt)} chars). "
            f"Connect cluster nodes for full inference.\n"
            f"Requested max_tokens={max_tokens}\n"
        )
        # Simulate streaming by yielding characters
        for char in response:
            yield char
            await asyncio.sleep(0.01)

    async def _distribute_prompt(self, prompt: str, nodes: list) -> list:
        """Tokenize and distribute prompt across nodes."""
        # Phase 2: implement actual tokenization + distribution
        return []

    async def _collect_logits(self, nodes: list) -> list:
        """Collect logits from the final node in pipeline."""
        # Phase 2: implement logit collection
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
