import logging
from fastapi import APIRouter, HTTPException
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
    total_ram = sum(n.get("memory", {}).get("total_bytes", 0) for n in nodes)
    total_gpus = sum(1 for n in nodes if n.get("gpus"))

    return {
        "status": "running",
        "nodes": len(nodes),
        "total_ram_gb": total_ram / (1024**3),
        "total_gpus": total_gpus,
        "models": discovery_service.get_available_models(),
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
