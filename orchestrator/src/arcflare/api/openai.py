import time
import uuid
import logging
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

logger = logging.getLogger("arcflare.api.openai")

router = APIRouter(tags=["OpenAI-compatible"])


# ─── Request/Response Models ───

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float = 0.7
    max_tokens: int = 1024
    stream: bool = False
    top_p: float = 1.0


class CompletionRequest(BaseModel):
    model: str
    prompt: str
    max_tokens: int = 1024
    temperature: float = 0.7
    stream: bool = False


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "arcflare"


class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelInfo]


# ─── API Endpoints ───

@router.get("/models")
async def list_models():
    from ..main import discovery_service
    if discovery_service is None:
        return ModelList(data=[ModelInfo(id="arcflare/default", created=int(time.time()))])

    models = discovery_service.get_available_models()
    return ModelList(data=[
        ModelInfo(id=m, created=int(time.time())) for m in (models or ["arcflare/default"])
    ])


@router.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if request.stream:
        return EventSourceResponse(generate_chat_stream(request))

    logger.info(f"Chat completion request: model={request.model}, messages={len(request.messages)}")

    from ..inference.pipeline import run_inference
    result = await run_inference(
        model=request.model,
        prompt=format_messages(request.messages),
        max_tokens=request.max_tokens,
        temperature=request.temperature,
    )

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": result,
            },
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


@router.post("/completions")
async def completions(request: CompletionRequest):
    if request.stream:
        return EventSourceResponse(generate_completion_stream(request))

    logger.info(f"Completion request: model={request.model}")
    from ..inference.pipeline import run_inference
    result = await run_inference(
        model=request.model,
        prompt=request.prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
    )

    return {
        "id": f"cmpl-{uuid.uuid4().hex[:12]}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [{
            "index": 0,
            "text": result,
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


async def generate_chat_stream(request: ChatCompletionRequest) -> AsyncGenerator[dict, None]:
    from ..inference.pipeline import run_inference_stream
    full_result = ""
    async for chunk in run_inference_stream(
        model=request.model,
        prompt=format_messages(request.messages),
        max_tokens=request.max_tokens,
        temperature=request.temperature,
    ):
        full_result += chunk
        yield {
            "event": "delta",
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "data": {
                "choices": [{
                    "index": 0,
                    "delta": {"content": chunk},
                    "finish_reason": None,
                }]
            },
        }
    yield {
        "event": "done",
        "data": "[DONE]",
    }


async def generate_completion_stream(request: CompletionRequest):
    from ..inference.pipeline import run_inference_stream
    async for chunk in run_inference_stream(
        model=request.model,
        prompt=request.prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
    ):
        yield {
            "event": "delta",
            "data": {
                "choices": [{
                    "index": 0,
                    "text": chunk,
                    "finish_reason": None,
                }]
            },
        }
    yield {
        "event": "done",
        "data": "[DONE]",
    }


def format_messages(messages: list[ChatMessage]) -> str:
    parts = []
    for msg in messages:
        if msg.role == "system":
            parts.append(f"System: {msg.content}")
        elif msg.role == "user":
            parts.append(f"User: {msg.content}")
        elif msg.role == "assistant":
            parts.append(f"Assistant: {msg.content}")
    return "\n".join(parts)
