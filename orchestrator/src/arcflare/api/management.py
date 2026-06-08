import logging
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

logger = logging.getLogger("arcflare.api.management")
router = APIRouter(tags=["Cluster Management"])


class NodeInfo(BaseModel):
    node_id: str
    name: str
    status: str
    address: str
    cpu_cores: int
    memory_gb: float
    has_gpu: bool
    temperature: float | None = None


class RegisterRequest(BaseModel):
    node_id: str
    name: str
    grpc_port: int = 9001
    rpc_port: int = 0
    version: str = "0.0.0"
    os: str = "unknown"
    # compact hardware summary, e.g. {"cpu_cores": 8, "ram_bytes": 167..., "gpu_count": 1}
    hardware: dict | None = None


@router.post("/nodes/register")
async def register_node(req: RegisterRequest, request: Request):
    from ..main import discovery_service
    from ..cluster.discovery import NodeInfo
    import time

    if discovery_service is None:
        raise HTTPException(503, "Discovery service not ready")

    # basic validation — reject empty identifiers and bad ports
    if not req.node_id.strip() or not req.name.strip():
        raise HTTPException(422, "node_id and name must be non-empty")
    if not (0 < req.grpc_port < 65536) or not (0 <= req.rpc_port < 65536):
        raise HTTPException(422, "ports out of range")

    client_ip = request.client.host if request.client else "127.0.0.1"
    existing = discovery_service.nodes.get(req.node_id)
    node_info = NodeInfo(
        node_id=req.node_id,
        node_name=req.name,
        grpc_port=req.grpc_port,
        rpc_port=req.rpc_port,
        version=req.version,
        os=req.os,
        ip_address=client_ip,
        last_seen=time.time(),
        status="alive",
        hardware=req.hardware if req.hardware is not None else (existing.hardware if existing else None),
    )
    discovery_service.nodes[req.node_id] = node_info
    logger.info(f"Node {'re-' if existing else ''}registered via HTTP: {req.name} ({req.node_id})"
                + (f" rpc={req.rpc_port}" if req.rpc_port else ""))
    return {"status": "registered", "node_id": req.node_id}


@router.get("/nodes")
async def list_nodes():
    from ..main import discovery_service
    if discovery_service is None:
        return {"nodes": []}
    return {"nodes": discovery_service.get_nodes()}


@router.get("/nodes/{node_id}")
async def get_node(node_id: str):
    from ..main import discovery_service
    if discovery_service is None:
        raise HTTPException(404, "Discovery service not ready")
    node = discovery_service.get_node(node_id)
    if node is None:
        raise HTTPException(404, f"Node {node_id} not found")
    return node


@router.get("/cluster/status")
async def cluster_status():
    from ..main import discovery_service
    if discovery_service is None:
        return {"status": "starting", "nodes": 0}

    nodes = discovery_service.get_nodes()
    # hardware is the compact summary supplied at registration (may be absent)
    total_ram = sum((n.get("hardware") or {}).get("ram_bytes", 0) for n in nodes)
    total_gpus = sum((n.get("hardware") or {}).get("gpu_count", 0) for n in nodes)
    degraded = sum(1 for n in nodes if n.get("status") == "degraded")
    rpc_endpoints = discovery_service.get_rpc_endpoints()

    return {
        "status": "running",
        "nodes": len(nodes),
        "degraded": degraded,
        "total_ram_gb": total_ram / (1024**3),
        "total_gpus": total_gpus,
        "models": discovery_service.get_available_models(),
        "rpc_endpoints": rpc_endpoints,
        "pipeline_mode": "rpc_distributed" if len(rpc_endpoints) >= 2 else (
            "single_rpc" if len(rpc_endpoints) == 1 else "local_fallback"
        ),
    }


@router.post("/cluster/tune")
async def tune_cluster():
    from ..cluster.partition import optimize_cluster
    result = await optimize_cluster()
    return {"status": "tuning_complete", "result": result}


@router.post("/cluster/benchmark")
async def benchmark_nodes():
    from ..main import discovery_service
    if discovery_service is None:
        return {"error": "Not ready"}
    results = await discovery_service.benchmark_all_nodes()
    return {"benchmarks": results}
