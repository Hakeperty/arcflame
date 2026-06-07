import asyncio
import json
import logging
import socket
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

logger = logging.getLogger("arcflare.discovery")

DISCOVERY_PORT = 5678
BROADCAST_ADDR = "255.255.255.255"
HEARTBEAT_TIMEOUT = 15  # seconds


@dataclass
class NodeInfo:
    node_id: str
    node_name: str
    grpc_port: int
    version: str
    os: str
    last_seen: float = 0.0
    hardware: Optional[dict] = None
    status: str = "discovered"
    ip_address: str = ""


class DiscoveryService:
    def __init__(self):
        self.nodes: Dict[str, NodeInfo] = {}
        self._running = False
        self._socket: Optional[socket.socket] = None

    async def start(self):
        self._running = True
        loop = asyncio.get_event_loop()

        # Create UDP socket for discovery
        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            self._socket.bind(("0.0.0.0", DISCOVERY_PORT))
            self._socket.setblocking(False)
            logger.info(f"Discovery listening on UDP port {DISCOVERY_PORT}")
        except OSError as e:
            logger.warning(f"Could not bind UDP discovery port: {e}. Nodes must register manually.")
            self._socket = None
            return

        # Reader task
        while self._running:
            try:
                data, addr = await loop.sock_recvfrom(self._socket, 8192)
                self._handle_discovery(data, addr)
            except (OSError, asyncio.TimeoutError):
                await asyncio.sleep(0.1)

        self._socket.close()

    def stop(self):
        self._running = False

    def _handle_discovery(self, data: bytes, addr: tuple):
        try:
            msg = json.loads(data.decode("utf-8"))
            node_id = msg.get("node_id", "")
            node_name = msg.get("node_name", "unknown")
            grpc_port = msg.get("grpc_port", 0)
            version = msg.get("version", "0.0.0")
            os_name = msg.get("os", "unknown")

            if node_id not in self.nodes:
                logger.info(f"New node discovered: {node_name} ({node_id}) at {addr[0]}")
                self.nodes[node_id] = NodeInfo(
                    node_id=node_id,
                    node_name=node_name,
                    grpc_port=grpc_port,
                    version=version,
                    os=os_name,
                    ip_address=addr[0],
                )
            else:
                self.nodes[node_id].last_seen = time.time()
                self.nodes[node_id].status = "alive"

        except (json.JSONDecodeError, KeyError) as e:
            logger.debug(f"Invalid discovery message from {addr}: {e}")

    def get_nodes(self) -> List[dict]:
        self._prune_dead_nodes()
        return [asdict(n) for n in self.nodes.values()]

    def get_node(self, node_id: str) -> Optional[dict]:
        node = self.nodes.get(node_id)
        return asdict(node) if node else None

    def get_available_models(self) -> List[str]:
        models = set()
        for node in self.nodes.values():
            if node.hardware and "models" in node.hardware:
                models.update(node.hardware["models"])
        return list(models) or ["arcflare/default"]

    def _prune_dead_nodes(self):
        now = time.time()
        dead = [
            nid for nid, n in self.nodes.items()
            if n.last_seen > 0 and (now - n.last_seen) > HEARTBEAT_TIMEOUT
        ]
        for nid in dead:
            logger.info(f"Node {self.nodes[nid].node_name} timed out")
            del self.nodes[nid]

    async def benchmark_all_nodes(self) -> dict:
        results = {}
        for node_id, node in self.nodes.items():
            if node.grpc_port > 0:
                results[node_id] = {
                    "name": node.node_name,
                    "status": "pending",
                }
        return results
