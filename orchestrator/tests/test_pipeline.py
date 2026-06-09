"""Unit tests for pipeline model resolution and discovery pruning."""
import os
import time

import pytest

from arcflare.inference.pipeline import InferencePipeline
from arcflare.cluster.discovery import DiscoveryService, NodeInfo, HEARTBEAT_TIMEOUT


def test_find_model_by_name(tmp_path, monkeypatch):
    (tmp_path / "tinyllama-1.1b.Q4_K_M.gguf").write_bytes(b"x")
    (tmp_path / "qwen2.5-0.5b.gguf").write_bytes(b"x")
    monkeypatch.setenv("ARCFLARE_MODELS_DIR", str(tmp_path))
    pl = InferencePipeline()

    # #11: name must actually be used, not "first .gguf found"
    assert pl._find_model("qwen2.5-0.5b").endswith("qwen2.5-0.5b.gguf")
    assert pl._find_model("tinyllama").endswith("tinyllama-1.1b.Q4_K_M.gguf")
    # namespaced + unknown falls back deterministically (sorted first)
    got = pl._find_model("arcflare/default")
    assert got.endswith(".gguf")


def test_find_model_missing_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("ARCFLARE_MODELS_DIR", str(tmp_path / "nope"))
    assert InferencePipeline()._find_model("x") is None


def test_discovery_prunes_silent_nodes():
    # #23: a discovered node must become prunable (last_seen stamped on first sight)
    svc = DiscoveryService()
    svc.nodes["n1"] = NodeInfo(
        node_id="n1", node_name="n1", grpc_port=9001, version="1", os="linux",
        rpc_port=10001, ip_address="10.0.0.2",
        last_seen=time.time() - HEARTBEAT_TIMEOUT - 5,  # gone silent
    )
    assert svc.get_rpc_endpoints() == []   # pruned
    assert "n1" not in svc.nodes


def test_get_rpc_endpoints_filters():
    svc = DiscoveryService()
    svc.nodes["a"] = NodeInfo("a", "a", 9001, "1", "linux", rpc_port=10001,
                              ip_address="10.0.0.2", last_seen=time.time())
    svc.nodes["b"] = NodeInfo("b", "b", 9002, "1", "linux", rpc_port=0,
                              ip_address="10.0.0.3", last_seen=time.time())  # no rpc
    eps = svc.get_rpc_endpoints()
    assert eps == ["10.0.0.2:10001"]
