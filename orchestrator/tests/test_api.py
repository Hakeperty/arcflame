"""API + regression tests for the orchestrator.

Covers the audit-fixed bugs so they can't silently regress:
- register validation (#24)
- cluster_status RAM/GPU from the hardware summary (#22)
- streaming SSE emits valid JSON (#21)
- dashboard serves
"""
import json

import pytest
from fastapi.testclient import TestClient

import arcflare.main as main
from arcflare.cluster.discovery import DiscoveryService


@pytest.fixture
def client(monkeypatch):
    # fresh discovery service per test, no UDP socket / lifespan needed
    main.discovery_service = DiscoveryService()
    return TestClient(main.app)


def test_health_and_root(client):
    assert client.get("/health").json()["status"] == "healthy"
    assert client.get("/")  .json()["dashboard"] == "/dashboard"


def test_dashboard_serves_html(client):
    r = client.get("/dashboard")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "ArcFlare" in r.text


def test_register_validation(client):
    # empty node_id -> 422
    assert client.post("/api/nodes/register", json={"node_id": "", "name": "x"}).status_code == 422
    # bad port -> 422
    assert client.post("/api/nodes/register",
                       json={"node_id": "a", "name": "a", "grpc_port": 0}).status_code == 422
    # valid -> registered
    r = client.post("/api/nodes/register", json={"node_id": "a", "name": "alpha"})
    assert r.status_code == 200 and r.json()["status"] == "registered"


def test_cluster_status_reports_hardware(client):
    # bug #22: RAM/GPU used to always be 0
    client.post("/api/nodes/register", json={
        "node_id": "n1", "name": "n1", "rpc_port": 10001,
        "hardware": {"cpu_cores": 4, "ram_bytes": 8 * 1024**3, "gpu_count": 1},
    })
    client.post("/api/nodes/register", json={
        "node_id": "n2", "name": "n2", "rpc_port": 10002,
        "hardware": {"cpu_cores": 8, "ram_bytes": 8 * 1024**3, "gpu_count": 0},
    })
    s = client.get("/api/cluster/status").json()
    assert s["nodes"] == 2
    assert s["total_ram_gb"] == pytest.approx(16.0)
    assert s["total_gpus"] == 1
    assert s["pipeline_mode"] == "rpc_distributed"
    assert sorted(s["rpc_endpoints"]) == ["testclient:10001", "testclient:10002"]


def test_pipeline_mode_transitions(client):
    assert client.get("/api/cluster/status").json()["pipeline_mode"] == "local_fallback"
    client.post("/api/nodes/register", json={"node_id": "n1", "name": "n1", "rpc_port": 10001})
    assert client.get("/api/cluster/status").json()["pipeline_mode"] == "single_rpc"


def test_chat_completions_nonstream(client, monkeypatch):
    async def fake_run(model, prompt, max_tokens, temperature):
        return "Paris."
    import arcflare.inference.pipeline as pl
    monkeypatch.setattr(pl, "run_inference", fake_run)

    r = client.post("/v1/chat/completions", json={
        "model": "arcflare/default",
        "messages": [{"role": "user", "content": "capital of France?"}],
    })
    assert r.status_code == 200
    body = r.json()
    assert body["choices"][0]["message"]["content"] == "Paris."
    assert body["object"] == "chat.completion"


def test_chat_completions_stream_is_valid_json(client, monkeypatch):
    # bug #21: each SSE data: payload must be valid JSON
    async def fake_stream(model, prompt, max_tokens, temperature):
        for tok in ["Hello", " ", "world"]:
            yield tok
    # openai.py imports run_inference_stream lazily from the pipeline module,
    # so patch it there.
    import arcflare.inference.pipeline as pl
    monkeypatch.setattr(pl, "run_inference_stream", fake_stream)

    with client.stream("POST", "/v1/chat/completions", json={
        "model": "arcflare/default",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    }) as r:
        assert r.status_code == 200
        payloads = []
        for line in r.iter_lines():
            if line.startswith("data:"):
                data = line[len("data:"):].strip()
                if data and data != "[DONE]":
                    payloads.append(json.loads(data))  # raises if not valid JSON
    assert payloads
    assert payloads[0]["object"] == "chat.completion.chunk"
    assert payloads[-1]["choices"][0]["finish_reason"] == "stop"
    text = "".join(p["choices"][0]["delta"].get("content", "") for p in payloads)
    assert text == "Hello world"
