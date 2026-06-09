"""Phase 3 crash-recovery / health-monitor tests."""
import asyncio
import socket
import time

import pytest

from arcflare.cluster.discovery import DiscoveryService, NodeInfo, HEALTH_FAIL_THRESHOLD


def _free_listening_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    s.listen(1)
    return s, s.getsockname()[1]


def test_probe_open_and_closed_ports():
    svc = DiscoveryService()
    sock, port = _free_listening_port()
    try:
        assert asyncio.run(svc._probe("127.0.0.1", port)) is True
    finally:
        sock.close()
    # nothing listening on this port now
    assert asyncio.run(svc._probe("127.0.0.1", port)) is False
    assert asyncio.run(svc._probe("", 0)) is False


def test_crashed_node_removed_after_threshold():
    svc = DiscoveryService()
    # node pointing at a dead port
    svc.nodes["n1"] = NodeInfo("n1", "n1", 9001, "1", "linux",
                               rpc_port=59999, ip_address="127.0.0.1",
                               last_seen=time.time())

    async def run():
        for _ in range(HEALTH_FAIL_THRESHOLD):
            await svc.check_all_nodes()
    asyncio.run(run())
    assert "n1" not in svc.nodes  # crashed node pruned


def test_busy_node_not_dropped_on_timeout(monkeypatch):
    # rpc-server has backlog 1; a busy node times out (probe -> None) but is alive.
    # It must NOT be removed and must stay fresh.
    svc = DiscoveryService()
    svc.nodes["n1"] = NodeInfo("n1", "n1", 9001, "1", "linux",
                               rpc_port=10001, ip_address="127.0.0.1",
                               consecutive_failures=2, last_seen=0.0)

    async def fake_probe(ip, port, timeout=2.0):
        return None  # inconclusive / busy
    monkeypatch.setattr(svc, "_probe", fake_probe)

    async def run():
        for _ in range(5):
            await svc.check_all_nodes()
    asyncio.run(run())
    assert "n1" in svc.nodes                       # survived
    assert svc.nodes["n1"].consecutive_failures == 0
    assert svc.nodes["n1"].last_seen > 0           # refreshed


def test_healthy_node_survives_and_resets_failures():
    svc = DiscoveryService()
    sock, port = _free_listening_port()
    try:
        svc.nodes["n1"] = NodeInfo("n1", "n1", 9001, "1", "linux",
                                   rpc_port=port, ip_address="127.0.0.1",
                                   consecutive_failures=2)

        async def run():
            return await svc.check_all_nodes()
        res = asyncio.run(run())
        assert res["n1"] is True
        assert svc.nodes["n1"].consecutive_failures == 0
        assert svc.nodes["n1"].status == "alive"
    finally:
        sock.close()
