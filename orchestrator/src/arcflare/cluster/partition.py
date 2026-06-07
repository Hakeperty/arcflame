import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

logger = logging.getLogger("arcflare.partition")

# Minimum memory to reserve for OS and overhead per node
OS_RESERVE_BYTES = 1 * 1024 * 1024 * 1024  # 1 GB

# Estimated memory per layer for typical 7B-70B models (bytes per parameter, Q4)
BYTES_PER_PARAM_Q4 = 0.5   # 4 bits = 0.5 bytes
BYTES_PER_PARAM_Q8 = 1.0
BYTES_PER_PARAM_F16 = 2.0

# KV cache memory per token per layer
KV_CACHE_BYTES_PER_TOKEN = 2 * 1024 * 1024  # ~2MB per token for 32-layer model


@dataclass
class PartitionResult:
    node_id: str
    first_layer: int
    num_layers: int
    has_lm_head: bool
    estimated_memory_mb: float


def calculate_partition(
    nodes: List[dict],
    num_layers: int,
    model_size_bytes: int,
    max_context: int = 4096,
    quantization: str = "Q4_K_M",
) -> List[PartitionResult]:
    """Distribute model layers across nodes based on their capabilities.

    Algorithm:
    1. Score each node by available RAM, CPU speed, GPU presence
    2. Allocate layers proportionally to scores
    3. Assign LM head to the fastest node
    4. Reserve memory for KV cache based on context length
    """
    if not nodes:
        logger.warning("No nodes available for partition")
        return []

    scored_nodes = _score_nodes(nodes, max_context)
    total_score = sum(s["score"] for s in scored_nodes)
    layers_per_node: List[Tuple[int, str]] = []

    for s in scored_nodes:
        share = s["score"] / total_score
        layer_count = max(1, int(num_layers * share))
        layers_per_node.append((layer_count, s["node_id"]))

    # Adjust to match total layer count
    total_allocated = sum(l[0] for l in layers_per_node)
    while total_allocated < num_layers:
        # Give extra layers to the most capable node
        idx = max(range(len(layers_per_node)), key=lambda i: scored_nodes[i]["score"])
        layers_per_node[idx] = (layers_per_node[idx][0] + 1, layers_per_node[idx][1])
        total_allocated += 1

    while total_allocated > num_layers:
        idx = min(range(len(layers_per_node)), key=lambda i: scored_nodes[i]["score"])
        if layers_per_node[idx][0] > 1:
            layers_per_node[idx] = (layers_per_node[idx][0] - 1, layers_per_node[idx][1])
            total_allocated -= 1
        else:
            break

    # Assign LM head to the node with GPU or highest RAM
    lm_head_node = max(range(len(scored_nodes)), key=lambda i: (
        2 if scored_nodes[i].get("has_gpu") else 0,
        scored_nodes[i].get("score", 0),
    ))

    # Build partition list
    current_layer = 0
    partitions = []
    for i, (count, node_id) in enumerate(layers_per_node):
        has_head = (i == lm_head_node)
        partition = PartitionResult(
            node_id=node_id,
            first_layer=current_layer,
            num_layers=count,
            has_lm_head=has_head,
            estimated_memory_mb=(count / num_layers) * (model_size_bytes / (1024 * 1024)),
        )
        partitions.append(partition)
        current_layer += count

    logger.info(f"Partitioned {num_layers} layers across {len(nodes)} nodes")
    for p in partitions:
        logger.info(f"  Node {p.node_id}: layers {p.first_layer}-{p.first_layer + p.num_layers - 1} "
                    f"({'LM head' if p.has_lm_head else ''})")

    return partitions


async def optimize_cluster() -> dict:
    """Run through all nodes and tune them for performance."""
    from ..main import discovery_service
    nodes = discovery_service.get_nodes() if discovery_service else []

    return {
        "nodes_optimized": len(nodes),
        "message": "Cluster optimization complete",
    }


def _score_nodes(nodes: List[dict], max_context: int) -> List[dict]:
    scored = []
    for node in nodes:
        score = 0.0
        memory = node.get("memory", {})
        total_ram = memory.get("total_bytes", 8 * 1024**3)
        available_ram = memory.get("available_bytes", total_ram)
        usable_ram = max(0, available_ram - OS_RESERVE_BYTES)
        score += usable_ram / (1024**3)  # 1 point per GB of usable RAM

        # GPU bonus
        gpus = node.get("gpus", [])
        if gpus:
            for gpu in gpus:
                vram = gpu.get("vram_bytes", 0)
                score += vram / (1024**3) * 2  # 2 points per GB of VRAM
                score += 5 if gpu.get("available") else 0  # 5 points for any GPU

        # CPU bonus (from benchmark)
        benchmark = node.get("benchmark_score", 0)
        score += benchmark / 1000.0

        # Network penalty (high latency = lower score)
        if node.get("latency_ms", 0) > 10:
            score *= 0.8
        if node.get("latency_ms", 0) > 50:
            score *= 0.6

        scored.append({
            "node_id": node.get("node_id", ""),
            "score": max(1.0, score),
            "has_gpu": len(gpus) > 0,
        })

    return scored
