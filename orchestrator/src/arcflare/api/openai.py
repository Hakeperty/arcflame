import json
import time
import uuid
import logging
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

# clamp generation length so a single request can't pin a worker indefinitely
MAX_TOKENS_LIMIT = 8192
from sse_starlette.sse import EventSourceResponse

logger = logging.getLogger("arcflare.api.openai")

router = APIRouter(tags=["OpenAI-compatible"])


# ─── Request/Response Models ───

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage] = Field(min_length=1)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, ge=1, le=MAX_TOKENS_LIMIT)
    stream: bool = False
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)


class CompletionRequest(BaseModel):
    model: str
    prompt: str = Field(min_length=1)
    max_tokens: int = Field(default=1024, ge=1, le=MAX_TOKENS_LIMIT)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
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
    # OpenAI streaming requires each SSE `data:` payload to be a JSON *string*.
    # sse_starlette renders dicts via str() (Python repr), which is invalid JSON
    # and breaks every OpenAI client — so we json.dumps() the chunk ourselves.
    cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    async for chunk in run_inference_stream(
        model=request.model,
        prompt=format_messages(request.messages),
        max_tokens=request.max_tokens,
        temperature=request.temperature,
    ):
        yield {"data": json.dumps({
            "id": cid,
            "object": "chat.completion.chunk",
            "created": created,
            "model": request.model,
            "choices": [{"index": 0, "delta": {"content": chunk}, "finish_reason": None}],
        })}
    # terminal chunk + sentinel
    yield {"data": json.dumps({
        "id": cid,
        "object": "chat.completion.chunk",
        "created": created,
        "model": request.model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    })}
    yield {"data": "[DONE]"}


async def generate_completion_stream(request: CompletionRequest):
    from ..inference.pipeline import run_inference_stream
    cid = f"cmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    async for chunk in run_inference_stream(
        model=request.model,
        prompt=request.prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
    ):
        yield {"data": json.dumps({
            "id": cid,
            "object": "text_completion",
            "created": created,
            "model": request.model,
            "choices": [{"index": 0, "text": chunk, "finish_reason": None}],
        })}
    yield {"data": json.dumps({
        "id": cid,
        "object": "text_completion",
        "created": created,
        "model": request.model,
        "choices": [{"index": 0, "text": "", "finish_reason": "stop"}],
    })}
    yield {"data": "[DONE]"}


def format_messages(messages: list[ChatMessage]) -> str:
    parts = []
    for msg in messages:
        role = msg.role.capitalize() if msg.role else "User"
        parts.append(f"{role}: {msg.content}")
    # prime the model to continue as the assistant
    parts.append("Assistant:")
    return "\n".join(parts)
