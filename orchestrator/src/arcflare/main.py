import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.openai import router as openai_router
from .api.management import router as management_router
from .api.dashboard import router as dashboard_router
from .cluster.discovery import DiscoveryService

logger = logging.getLogger("arcflare")

discovery_service: DiscoveryService | None = None
_background_tasks: set = set()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global discovery_service

    logger.info("Starting ArcFlare orchestrator...")
    discovery_service = DiscoveryService()
    # keep strong references — a bare create_task() can be garbage-collected
    # mid-flight, silently cancelling the task and swallowing its errors
    for coro in (discovery_service.start(), discovery_service.health_loop()):
        task = asyncio.create_task(coro)
        _background_tasks.add(task)
        task.add_done_callback(_background_tasks.discard)
    logger.info("Discovery + health monitor started")

    yield

    logger.info("Shutting down ArcFlare orchestrator...")
    if discovery_service:
        discovery_service.stop()
    logger.info("Shutdown complete")


app = FastAPI(
    title="ArcFlare",
    description="Distributed LLM inference for old/scrap hardware",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(openai_router, prefix="/v1")
app.include_router(management_router, prefix="/api")
app.include_router(dashboard_router)


@app.get("/")
async def root():
    return {
        "name": "ArcFlare",
        "version": "0.1.0",
        "status": "running",
        "dashboard": "/dashboard",
        "docs": "/docs",
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}
