# ArcFlare Phase 3 — Full-Stack 3-VM Validation (real node-agent)

**Date:** 2026-06-09
**Branch:** `phase3-maintenance`
**Goal:** validate the whole system on three separate KVM VMs using the **real
compiled `node-agent`** (the earlier `PHASE2_VM_TEST_REPORT.md` had to fake
registration with curl — no agent binary existed then).

---

## Result — PASS

A 3-VM cluster ran end-to-end with the actual Rust agent: agents auto-registered
with hardware, the orchestrator served distributed inference via the persistent
`llama-server`, and a node crash was detected and recovered from automatically.

## Topology
- `orch` (10.10.0.1) — orchestrator (new code from this branch) + persistent `llama-server`
- `node-a` (10.10.0.2) — real `node-agent --enable-rpc` (started its own `rpc-server`)
- `node-b` (10.10.0.3) — real `node-agent --enable-rpc`

Agents run as systemd transient units (`systemd-run --unit=arcflare-node`) so they
survive the provisioning SSH session. The compiled `node-agent` (GLIBC_2.39 max)
runs unmodified on the Ubuntu 24.04 guests, shared in over 9p.

## What was validated

### 1. Real agent startup + auto-registration
`node-a` agent log:
```
Hardware detected on node-a: 2 cores, 877789184 RAM
llama rpc-server started on 0.0.0.0:10001
Starting UDP discovery broadcaster on port 5678
Registered with orchestrator at 10.10.0.1:8000
Node agent starting on 0.0.0.0:9001
```
Both agents auto-registered. `/api/cluster/status`:
```
mode=rpc_distributed nodes=2 RAM=1.64GB gpus=0
eps=['10.10.0.2:10001','10.10.0.3:10001']
```
✅ **RAM now reported (1.64GB)** — the agent sends a hardware summary on register
and the orchestrator sums it (bug #22, validated with the real agent, not a curl stub).

### 2. node_id collision fix
Both VMs are clones of one base image → identical machine-id. With the old
`node_id = machine_uid + grpc_port`, both registered as the *same* node (only one
showed up). After adding the hostname to `node_id`, both appear. This is a real
bug for "flash one SD card, clone it" scrap-hardware fleets.

### 3. Distributed inference via persistent llama-server
| | latency | answer |
|---|---|---|
| cold (model load + ship tensors to both VMs over the virtual LAN) | 42s | "The capital of France is Paris." |
| warm (model stays resident) | **2s** | "The color of the ocean." |

✅ The persistent-server optimization holds across real VMs: warm ~2s vs the old
~23s per-request reload.

### 4. Crash recovery
Stopped `node-b`'s agent (`systemctl stop arcflare-node` → kills agent + its rpc-server):
```
t=15s: nodes=2 degraded=1
t=20s: nodes=1 single_rpc eps=['10.10.0.2:10001']   ← node-b dropped
inference after crash: "8 plus 1 = 9"               ← recovered via survivor
```
✅ The health monitor detected the crash and the persistent `llama-server`
reconfigured to the surviving node — inference kept working.

## Bugs found & fixed during this test
- **node-agent startup panic** (clap `-o` collision) — fixed earlier; the agent
  now starts at all.
- **node_id collision on cloned images** — fixed (hostname in node_id).
- **Health probe false-negatives on busy rpc-servers** (backlog=1) — fixed
  earlier with the tri-state probe; confirmed here (busy survivor not dropped).

## Conclusion
The earlier VM report proved the *transport* worked with a stubbed registration.
This one proves the **whole stack** — real Rust agent, hardware auto-reporting,
UDP discovery, persistent distributed `llama-server`, and crash recovery — works
across three independent machines. Combined with the 15-test unit suite and the
clean `cargo build`, Phase 3 (dashboard + crash recovery) and the Phase 2
optimization are validated end-to-end.
