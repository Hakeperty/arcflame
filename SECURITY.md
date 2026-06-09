# ArcFlare — Security Model & Hardening

ArcFlare is designed for a **trusted LAN** of machines you control. Read this
before exposing any component to an untrusted network.

## Trust model

- The orchestrator HTTP API (`:8000`) and the node-agent gRPC server (`:9001`)
  are **unauthenticated**. Anyone who can reach those ports can list nodes, run
  inference, and register nodes.
- The llama.cpp `rpc-server` is bound to `0.0.0.0` so the orchestrator can reach
  it. llama.cpp itself warns this protocol is **not secure** and must not be
  exposed to an open network — a malicious RPC peer can compromise the process.

**Do not expose ports 8000 / 9001 / the rpc ports (10001+) to the internet.**
Keep them on a private LAN/VPN, or put an authenticating reverse proxy in front
of the orchestrator.

## Privileged operations are opt-in (default OFF)

The node-agent can apply CPU/system tuning and **aggressive overclocking /
undervolting** (`intel-undervolt`, `ryzenadj`, `nvidia-smi` power/clock). Because
the gRPC API is unauthenticated, these are **refused by default** — a LAN peer
cannot retune or overclock your hardware unless you explicitly opt in at launch:

| flag | enables | risk |
|---|---|---|
| (none) | nothing — `SetPerformanceMode` / `ApplySystemTuning` return `PERMISSION_DENIED` | — |
| `--allow-tuning` | governor, swappiness, hugepages, I/O scheduler, THP, NUMA | reversible system tuning |
| `--allow-aggressive` | CPU undervolt, raised TDP, GPU power-limit/clock | **can destabilize or damage hardware** |

Only pass `--allow-aggressive` on machines you own and can monitor.

## Input limits

- `max_tokens` is clamped to `[1, 8192]` and prompts/messages must be non-empty,
  so a single request can't pin a worker indefinitely.
- Node registration validates `node_id`/`name` (non-empty) and ports (range).

## Notes for operators

- Subprocesses are spawned without a shell (`exec`-style), so prompts/model
  names cannot inject shell commands.
- Tuning writes go to fixed `/proc` and `/sys` paths (no path injection) and
  require root to take effect.
- A node's advertised RPC endpoint is the IP it registers from; treat node
  registration as trusted (anyone on the LAN can register a worker the
  orchestrator will send model tensors to).

## Reporting

This is a hobby/scrap-hardware project. If you find a vulnerability, open an
issue describing the impact and reproduction.
