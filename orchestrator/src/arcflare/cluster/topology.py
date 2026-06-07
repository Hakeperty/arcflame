import logging
from typing import Dict, List, Optional

logger = logging.getLogger("arcflare.topology")


class TopologyManager:
    """Manages the cluster topology: node connectivity, latency, bandwidth."""

    def __init__(self):
        self.links: Dict[str, Dict[str, LinkStats]] = {}

    def record_latency(self, from_node: str, to_node: str, latency_ms: float):
        self.links.setdefault(from_node, {})[to_node] = LinkStats(
            latency_ms=latency_ms
        )

    def get_latency(self, from_node: str, to_node: str) -> Optional[float]:
        if from_node in self.links and to_node in self.links[from_node]:
            return self.links[from_node][to_node].latency_ms
        return None

    def get_fastest_path(self, start: str, end: str) -> List[str]:
        """Simple path finding based on lowest cumulative latency."""
        visited = set()
        path = []
        current = start

        while current != end:
            visited.add(current)
            neighbors = self.links.get(current, {})
            best_next = None
            best_latency = float("inf")

            for node, stats in neighbors.items():
                if node not in visited and stats.latency_ms < best_latency:
                    best_next = node
                    best_latency = stats.latency_ms

            if best_next is None:
                break

            path.append(best_next)
            current = best_next

        return path

    def to_dict(self) -> dict:
        return {
            node: {neighbor: s.to_dict() for neighbor, s in links.items()}
            for node, links in self.links.items()
        }


class LinkStats:
    def __init__(self, latency_ms: float = 0.0, bandwidth_mbps: float = 0.0):
        self.latency_ms = latency_ms
        self.bandwidth_mbps = bandwidth_mbps

    def to_dict(self) -> dict:
        return {
            "latency_ms": self.latency_ms,
            "bandwidth_mbps": self.bandwidth_mbps,
        }
