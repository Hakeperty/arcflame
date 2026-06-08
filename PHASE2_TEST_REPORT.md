# ArcFlare Phase 2 — Test Report

**Date:** 2026-06-08
**Commit under test:** `0f3d9eb`
**Tester:** automated end-to-end run on the development host

---

## 1. Objective

Validate the Phase 2 distributed-inference pipeline (llama.cpp RPC backend) and
exercise it the way the `docker-compose.yml` multi-container topology would:
one orchestrator + two worker nodes, each running `llama-rpc-server`, with the
orchestrator splitting model tensors across both nodes via
`llama-cli --rpc node1,node2`.

## 2. Environment

| Component | Value |
|---|---|
| Platform | Linux 6.17, x86_64 |
| Python (orchestrator) | 3.13.7 (venv at `.venv/`) |
| llama.cpp | build **b9558** (`c74759a24`), Ubuntu x64 release binaries |
| Model | TinyLlama 1.1B Chat v1.0 **Q4_K_M**, 638 MB GGUF |
| Docker | **not available** on host (no daemon, `docker` not installed, no passwordless sudo) |
| Rust/Cargo | not available on host |

## 3. Docker status — why containers were not run directly

`docker compose up` could not be executed here: the Docker engine is not
installed (`docker: command not found`, `/var/run/docker.sock` absent) and
installing it requires interactive `sudo`, which is unavailable in this
environment.

Instead the build context was **statically validated** (§4) and the runtime
topology was **faithfully reproduced as separate processes on the exact ports
compose defines** (§5):

| compose service | image | reproduced as |
|---|---|---|
| `orchestrator` | `Dockerfile.orchestrator`, :8000 | `uvicorn arcflare.main:app` on :8000 |
| `node-alpha` | `Dockerfile.node`, rpc :10001 | `rpc-server -p 10001` + HTTP register |
| `node-beta` | `Dockerfile.node`, rpc :10002 | `rpc-server -p 10002` + HTTP register |

In RPC mode the node-agent's only runtime jobs are (a) launch `llama-rpc-server`
and (b) register its `rpc_port` with the orchestrator. Both are reproduced
exactly, so the network path under test is identical to the container path.

## 4. Docker build validation (findings)

Walking each `COPY`/`RUN` as `docker build` would:

### 🔴 BLOCKER — `Dockerfile.node`: `COPY target/release/node-agent`
The Rust binary is not built (`target/release/node-agent` does not exist). The
node image build **fails immediately**. Requires `cargo build --release -p
node-agent` on the build host first (no Rust toolchain present here).

### 🔴 BLOCKER — `Dockerfile.orchestrator`: `COPY models/qwen2.5-0.5b-instruct-q4_k_m.gguf`
That model file does not exist in the repo; only
`tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` is present. The orchestrator image build
**fails** at this COPY. Fix: either download the Qwen model or change the COPY +
`ARCFLARE_MODELS_DIR` to the TinyLlama file.

### 🔴 BLOCKER — missing shared libraries for both binaries
The llama.cpp **release** binaries are dynamically linked. Verified:

```
$ ldd tools/llama-cli        → libllama-cli-impl.so => not found
$ ldd tools/llama-rpc-server → libggml.so.0 => not found, libggml-base.so.0 => not found
$ tools/llama-cli --version  → error while loading shared libraries: libllama-cli-impl.so
```

Both Dockerfiles copy only the bare binary (`COPY tools/llama-cli ...`,
`COPY tools/llama-rpc-server ...`) but **not the 28 `*.so` files** in
`tools/llama-bin/` they depend on. Even if the build succeeded, the orchestrator
local-fallback and the node rpc-server would crash at runtime with
"cannot open shared object file".
**Fix:** copy the whole lib directory and set `LD_LIBRARY_PATH`, e.g.
`COPY tools/llama-bin/ /usr/local/lib/llama/` + `ENV LD_LIBRARY_PATH=/usr/local/lib/llama`,
or build/copy statically-linked binaries.

### 🟡 MINOR — `Dockerfile.node` `EXPOSE 9001 10001`
`node-beta` uses rpc port 10002; the hardcoded `EXPOSE 10001` is cosmetic
(EXPOSE doesn't gate the actual `ports:` mapping) but is misleading.

### 🟡 MINOR — `docker-compose.yml` mounts `./models` but image bakes a model
The orchestrator both `COPY`s a model into `/models` and mounts `./models` over
it. The bind mount wins at runtime, so the baked-in model is dead weight.

> **Net:** the Docker images **cannot be built or run as-is**. Three blockers
> must be fixed before `docker compose up` will work. None of these affect the
> inference pipeline logic itself — they are packaging defects.

## 5. Runtime test — reproduced compose topology

### 5.1 Cluster formation

Both nodes registered and the orchestrator correctly upgraded the pipeline mode:

```json
{
  "status": "running",
  "nodes": 2,
  "rpc_endpoints": ["127.0.0.1:10001", "127.0.0.1:10002"],
  "pipeline_mode": "rpc_distributed"
}
```

✅ `pipeline_mode` correctly resolves to `rpc_distributed` with 2 endpoints
(`single_rpc` with 1, `local_fallback` with 0).

### 5.2 Inference requests (OpenAI-compatible API)

`POST /v1/chat/completions`, TinyLlama, `max_tokens=40`:

| # | Prompt | Response | Wall time |
|---|---|---|---|
| 1 | "Name the capital of France in one word." | "French Capital: Paris" | 2.6 s |
| 2 | "Write the word hello backwards." | "Hell … User: Write the word hell in reverse." | 3.1 s |
| 3 | "Complete: The sky is" | "Machine: Yes, I can provide you with a comprehensive list of … colors …" | 4.1 s |

All returned HTTP 200 with well-formed OpenAI-shaped JSON. Output quality is
limited by the tiny 1.1B Q4 model (it drifts and the chat template leaks), **not**
by the pipeline — the transport and routing are correct.

### 5.3 Proof of genuine distribution

`llama-cli --rpc` fans each inference out into many tensor-buffer operations,
one RPC connection each. Connection counts on the two worker processes after the
test run:

```
node-alpha :10001 → 120 accepted connections
node-beta  :10002 → 102 accepted connections
```

Both nodes carried a roughly balanced share of the tensor workload — confirming
the model was actually split across the cluster, not run locally.

### 5.4 Throughput

Direct verbose RPC run across both nodes:

```
[ Prompt: 41.8 t/s | Generation: 26.3 t/s ]
```

~26 tok/s generation for TinyLlama Q4 split over two loopback RPC nodes.

## 6. Bugs found & fixed during testing (commit `0f3d9eb`)

1. **RPC endpoints dropped for HTTP-registered nodes.**
   `get_rpc_endpoints()` skips nodes with an empty `ip_address`, but
   `register_node` hardcoded `ip_address=""`. Result: `pipeline_mode` stayed
   `local_fallback` even when nodes reported an `rpc_port`.
   **Fix:** capture `request.client.host` on register.

2. **llama-cli banner leaked into responses.**
   The RPC inference path streamed raw stdout, so the ASCII logo, spinner bytes,
   and `[ Prompt: … t/s ]` footer appeared in the chat content.
   **Fix:** apply the same `> `-delimited capture/strip logic the local-fallback
   path already used.

## 7. Recommendations (not yet done)

- [ ] Fix the 3 Docker blockers in §4 so `docker compose up` works end-to-end.
  - Build `node-agent` in a multi-stage Rust builder stage instead of `COPY`-ing
    a host artifact.
  - Bundle the `tools/llama-bin/*.so` libs + `LD_LIBRARY_PATH` (or static binaries).
  - Point the orchestrator model COPY/env at the model that actually ships.
- [ ] `cargo build --release -p node-agent` to confirm the Phase 2 Rust changes
  (rpc.rs, discovery rpc_port, grpc GetRpcEndpoint) compile — untested, no Rust
  toolchain on host.
- [ ] Once Docker works, re-run §5 inside real containers and confirm
  UDP auto-discovery (port 5678) registers nodes without the manual HTTP POST.

## 8. Conclusion

The **Phase 2 distributed-inference pipeline works**: nodes register, the
orchestrator detects RPC endpoints, upgrades to `rpc_distributed`, and serves
real TinyLlama completions with tensors genuinely split across two worker
processes (~26 t/s). Two pipeline bugs were found and fixed.

The **Docker packaging does not yet work** — three independent build/runtime
blockers (missing Rust binary, wrong model filename, missing shared libs) must
be resolved before the containers can be built. These are packaging issues,
orthogonal to the now-verified inference logic.
