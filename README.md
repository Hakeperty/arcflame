# ArcFlare

**Distributed LLM inference across scrap hardware.** Chain old laptops, Raspberry Pis, and desktops into a single AI cluster. ArcFlare splits a model across your devices so you can run models larger than any single machine can handle.

## How It Works

```
┌─────────────────────────────────────────────────────┐
│                     Your LAN                          │
│                                                      │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐       │
│  │ Orchestr │    │  Node A  │    │  Node B  │       │
│  │  Python   │◄──►│  Rust    │◄──►│  Rust    │       │
│  │  API +    │    │ gRPC    │    │ gRPC    │       │
│  │  Control  │    │ Server   │    │ Server   │       │
│  └────┬─────┘    └──────────┘    └──────────┘       │
│       │                                               │
│       └── Your apps (curl, OpenWebUI, Claude Code)    │
└─────────────────────────────────────────────────────┘
```

1. **Each machine** runs a lightweight `node-agent` (Rust) or orchestrator (Python)
2. **Nodes auto-discover** each other via UDP broadcast — no config needed
3. **Orchestrator** coordinates inference, splitting model layers across nodes
4. **You interact** via OpenAI-compatible API at `http://<orchestrator>:8000`

## Installation (Step by Step)

### Step 1 — Clone the repo

```bash
git clone https://github.com/Hakeperty/arcflare.git
cd arcflare
```

### Step 2 — Install prerequisites

**On every machine** (orchestrator + all nodes):

```bash
# Linux (Ubuntu/Debian)
sudo apt install cmake build-essential clang libclang-dev
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain nightly
source "$HOME/.cargo/env"
```

**MacOS:**

```bash
brew install cmake llvm
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain nightly
source "$HOME/.cargo/env"
```

**Raspberry Pi (ARM):**

```bash
sudo apt install cmake build-essential clang libclang-dev
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain nightly
source "$HOME/.cargo/env"
```

### Step 3 — Install Python dependencies (orchestrator machine only)

```bash
pip install -r orchestrator/requirements.txt
```

### Step 4 — Build the node agent (every machine)

```bash
cargo build --release -p node-agent
sudo cp target/release/node-agent /usr/local/bin/arcflare-node
```

### Step 5 — Download a model (orchestrator machine)

ArcFlare uses GGUF format models (see [llama.cpp](https://github.com/ggml-org/llama.cpp)):

```bash
mkdir -p models

# Small model for testing (469MB)
pip install huggingface-hub
python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='Qwen/Qwen2.5-0.5B-Instruct-GGUF',
    filename='qwen2.5-0.5b-instruct-q4_k_m.gguf',
    local_dir='models',
)
"
```

### Step 6 — Install `llama-cli` (orchestrator machine)

```bash
curl -sL "https://github.com/ggml-org/llama.cpp/releases/download/b9547/llama-b9547-bin-ubuntu-x64.tar.gz" \
  | tar -xz --strip=1 -C /usr/local/bin '*/llama-cli'
llama-cli --version  # verify
```

### Step 7 — Start the orchestrator (pick one machine)

```bash
cd orchestrator/src
ARCFLARE_LLAMA_CLI=/usr/local/bin/llama-cli \
ARCFLARE_MODELS_DIR=/home/$USER/arcflare/models \
uvicorn arcflare.main:app --host 0.0.0.0 --port 8000
```

### Step 8 — Join nodes to the cluster

**On each worker machine (basic):**

```bash
arcflare-node --orchestrator-host <orchestrator-ip>
```

**With RPC pipeline parallelism enabled (Phase 2):**

Each node runs `llama-rpc-server` alongside the agent. The orchestrator then
sends inference requests via `llama-cli --rpc <node1:port>,<node2:port>,...`
so model tensors are split across all machines.

Build `llama-rpc-server` once from llama.cpp with `LLAMA_RPC=ON`:

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build -DLLAMA_RPC=ON
cmake --build build --config Release -j$(nproc)
cp build/bin/llama-rpc-server /usr/local/bin/
```

Then start each node with `--enable-rpc`:

```bash
arcflare-node \
    --orchestrator-host <orchestrator-ip> \
    --enable-rpc \
    --rpc-port 10001 \
    --rpc-server-bin /usr/local/bin/llama-rpc-server
```

The orchestrator detects rpc endpoints automatically and switches to RPC mode.

**On the orchestrator machine itself** (if also running a node):

```bash
arcflare-node --orchestrator-host 127.0.0.1 --grpc-port 9001 --name orchest-node
```

### Step 9 — Verify the cluster

```bash
curl http://<orchestrator-ip>:8000/api/cluster/status
```

Expected output — you should see all machines listed under `nodes`.

### Step 10 — Chat!

```bash
curl -X POST http://<orchestrator-ip>:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"arcflare/default","messages":[{"role":"user","content":"Hello!"}]}'
```

---

### Alternative: Docker (single-machine test)

```bash
docker compose up -d
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"arcflare/default","messages":[{"role":"user","content":"Hello!"}]}'
```

### Alternative: Local dev cluster (all on one machine)

```bash
# Terminal 1 — orchestrator
cd orchestrator/src
ARCFLARE_LLAMA_CLI=/usr/local/bin/llama-cli \
ARCFLARE_MODELS_DIR=/home/$USER/arcflare/models \
uvicorn arcflare.main:app --host 0.0.0.0 --port 8000

# Terminal 2 — node alpha
arcflare-node --orchestrator-host 127.0.0.1 --grpc-port 9001 --name alpha

# Terminal 3 — node beta
arcflare-node --orchestrator-host 127.0.0.1 --grpc-port 9002 --name beta
```

## API

ArcFlare is **OpenAI API compatible**. Use any OpenAI client/tool:

```bash
# List models
curl http://localhost:8000/v1/models

# Chat
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"arcflare/default","messages":[{"role":"user","content":"Hello"}],"stream":true}'

# Completions
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"arcflare/default","prompt":"Once upon a time","max_tokens":100}'
```

### Management API

```bash
# Cluster status
curl http://localhost:8000/api/cluster/status

# List registered nodes
curl http://localhost:8000/api/nodes

# Node details
curl http://localhost:8000/api/nodes/<node-id>
```

### OpenAI-compatible integration

ArcFlare works as a drop-in OpenAI backend for any tool.

## Pipeline Parallelism (Phase 2)

When nodes run with `--enable-rpc`, the orchestrator automatically detects
their `llama-rpc-server` endpoints and upgrades from single-node inference to
true distributed inference:

```
Request → orchestrator (llama-cli)
                ├─── --rpc node-alpha:10001  (layers 0..N/2)
                └─── --rpc node-beta:10002   (layers N/2..N)
```

The orchestrator picks the inference mode in this order:

1. **RPC distributed** — if any node reports an rpc_port, all rpc endpoints are
   passed to `llama-cli --rpc`. Model tensors are split automatically.
2. **gRPC streaming** — falls back to sending the full prompt to one node via
   the ArcFlare gRPC ForwardStream protocol.
3. **Local fallback** — runs `llama-cli` directly on the orchestrator.

The active mode is reported in `/api/cluster/status` as `pipeline_mode`.

### Persistent model server

By default the orchestrator runs a single long-lived `llama-server` rather than
spawning a fresh `llama-cli` per request. The model is loaded **once** (and, in
RPC mode, its tensors are shipped to the nodes once); subsequent requests just
stream tokens over its local HTTP API. This cut warm-request latency from ~23s
(reload every time) to ~1.4s in local testing. Set `ARCFLARE_LLAMA_SERVER` to
the `llama-server` binary (or it's auto-discovered next to `llama-cli`); the
server restarts automatically when the model or the set of RPC nodes changes.

## Web Dashboard (Phase 3)

Open `http://<orchestrator>:8000/dashboard` for a live view of the cluster:
status, total RAM/GPUs, `pipeline_mode`, the node table (with RPC endpoints),
and an inference tester. Pure HTML/JS, no build step, refreshes every 3s.

## Health Monitoring & Crash Recovery (Phase 3)

The orchestrator actively probes each node every 10s. A node whose port is
refused for 3 consecutive checks is dropped; a busy-but-alive node (its probe
times out while serving) is kept. When a node disappears, the persistent
`llama-server` is automatically reconfigured to use the remaining nodes — a
crashed worker degrades the cluster instead of breaking it. `/api/cluster/status`
reports a `degraded` count.

## Auto-Discovery

Nodes on the same LAN automatically find each other:

1. Each node broadcasts a UDP heartbeat on port **5678** every 5 seconds
2. The orchestrator listens and registers new nodes automatically
3. Nodes also register via HTTP POST to `http://<orchestrator>:8000/api/nodes/register`
4. No manual IP configuration needed — just run the agent

Discovery message includes:
- Node ID (machine UID + port)
- Node name
- gRPC port
- OS and version

## Hardware Detection

Each node agent reports its hardware to the orchestrator:

- **CPU**: cores, architecture, frequency
- **RAM**: total and available
- **GPU**: vendor, model, VRAM
- **Drivers**: NVIDIA/CUDA status, Vulkan support
- **Benchmark**: CPU performance score

The orchestrator uses this data to decide how to split model layers across nodes.

## Overclocking & Tuning

Built-in performance optimization for scrap hardware:

| Mode | What it does |
|---|---|
| **Safe** (default) | Governor tweaks, hugepages, IO scheduler |
| **Aggressive** | Overclock CPU, undervolt GPU, max fans |
| **Driver audit** | Checks for outdated GPU drivers |
| **System tuning** | Kernel params, swap, NUMA balancing |

```bash
# Via API
curl -X POST http://localhost:8000/api/nodes/<id>/tune
curl -X POST http://localhost:8000/api/nodes/<id>/benchmark
```

## Architecture

```
arcflare/
├── node-agent/          # Rust — runs on each node
│   ├── src/
│   │   ├── hardware/    # CPU/GPU/RAM detection
│   │   ├── overclocking/# Safe + aggressive tuning
│   │   ├── drivers/     # GPU driver audit
│   │   ├── tuning/      # System optimization
│   │   ├── inference/   # Model loading + forward pass
│   │   └── network/     # gRPC server + UDP discovery
│   └── Cargo.toml
├── orchestrator/        # Python — cluster brain
│   ├── src/arcflare/
│   │   ├── api/         # OpenAI-compatible + management API
│   │   ├── cluster/     # Discovery, partitioning
│   │   ├── inference/   # Pipeline coordinator
│   │   └── models/      # Download + shard management
│   └── pyproject.toml
├── proto/               # gRPC service definitions
└── tools/
    └── gguf-splitter/   # GGUF analysis + partition planner
```

## Development

```bash
# Build everything
cargo build --release -p node-agent

# Run tests
pip install -r orchestrator/requirements-dev.txt
python3 -m pytest orchestrator/tests/      # 15 tests, no GPU/model needed
cargo build -p node-agent                  # node-agent compiles (default features)

# Local multi-node test
bash tools/scripts/test-cluster.sh

# Docker multi-node test
bash tools/scripts/test-docker.sh
```

## Roadmap

### Phase 1 ✅ (Done)
- [x] Project scaffolding
- [x] gRPC protocol definition
- [x] Node agent: hardware detection, gRPC server
- [x] UDP discovery
- [x] Orchestrator: FastAPI scaffold, node registry
- [x] GGUF splitter
- [x] Overclocking (safe + aggressive)
- [x] Driver auditing
- [x] System tuning
- [x] OpenAI-compatible API
- [x] CLI integration adapters
- [x] Docker multi-node cluster
- [x] Real inference via llama-cli

### Phase 2 ✅ (mostly done)
- [x] Full distributed inference pipeline — llama.cpp RPC backend
- [x] Persistent `llama-server` (model loaded once; ~23s → ~1.4s warm requests)
- [ ] P2P shard transfer (libp2p)
- [ ] Custom layer partitioning engine
- [ ] llama-cpp-4 inference integration
- [ ] KV cache optimization
- [ ] Quantizing and training opportunity

### Phase 3 🔜
- [x] Web dashboard (`/dashboard`)
- [x] Crash recovery — active health monitoring, auto-drop dead nodes,
      llama-server reconfigures around survivors
- [ ] Windows node agent support
- [ ] Live re-partitioning

## License

MIT

