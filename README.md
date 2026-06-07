# ArcFlare

**Distributed LLM inference for old/scrap hardware.**

Turn old laptops (ThinkPads, Dells, Optiplexes), Raspberry Pis, and other scrap devices into a unified cluster for running large language models.

## Architecture

```
┌────────────────────────────────────────┐
│           Orchestrator (Python)         │
│  Manages cluster, provides OpenAI API  │
└──────────────┬─────────────────────────┘
               │ gRPC control
     ┌─────────┼─────────┐
     │         │         │
┌────▼────┐ ┌──▼──┐ ┌───▼────┐
│ Node    │ │Node │ │ Node   │
│ Agent   │ │Agent│ │ Agent  │
│ (Rust)  │ │     │ │        │
│         │ │     │ │        │
│ • HW    │ │ ... │ │ ...    │
│ • OC    │ │     │ │        │
│ • Inf   │ │     │ │        │
└─────────┘ └─────┘ └────────┘
```

- **Orchestrator** (Python/FastAPI): Controls the cluster, provides OpenAI-compatible API
- **Node Agent** (Rust): Runs on each device — optimizes hardware, runs inference via llama.cpp

## Features

- **Distributed inference**: Split large models (30B-236B+) across multiple low-RAM devices
- **Pipeline parallelism**: Each node processes a subset of layers, passes hidden states to the next
- **GGUF sharding**: Split models into per-layer shards for efficient distribution
- **Auto-overclocking**: Safe mode (governor, hugepages) + Aggressive mode (undervolt, TDP, GPU OC)
- **Driver auditing**: Check NVIDIA/AMD/Intel driver status, get upgrade recommendations
- **System tuning**: Swappiness, hugepages, I/O scheduler, NUMA balancing, THP
- **CLI integration**: OpenAI-compatible API works with OpenCode, Claude Code, Qwen Code
- **P2P model distribution**: Nodes share model shards between each other

## Quick Start

### Prerequisites
- Rust toolchain (for node agent)
- Python 3.11+ (for orchestrator)

### Build

```bash
# Build the node agent
cargo build --release -p node-agent

# Build the GGUF splitter
cargo build --release -p gguf-splitter

# Install Python dependencies
cd orchestrator && pip install -r requirements.txt
```

### Run (single machine development mode)

```bash
# Terminal 1: Start orchestrator
cd orchestrator && uvicorn arcflare.main:app --reload --port 8000

# Terminal 2: Start simulated nodes
./target/release/node-agent --grpc-port 9001 --name node-alpha
./target/release/node-agent --grpc-port 9002 --name node-beta
./target/release/node-agent --grpc-port 9003 --name node-gamma
```

### Deploy to real hardware

```bash
# Deploy node agent to a remote machine
./tools/scripts/deploy-node.sh user@old-laptop.local
```

### Use with CLI tools

```bash
# OpenCode
OPENCODE_CONFIG_CONTENT='{"provider":{"id":"arcflare","model":"arcflare/default","urls":{"base_url":"http://localhost:8000/v1"}}}' opencode

# Claude Code
ANTHROPIC_BASE_URL=http://localhost:8000 ANTHROPIC_API_KEY=arcflare-dev-key claude

# Qwen Code / any OpenAI-compatible
OPENAI_BASE_URL=http://localhost:8000/v1 OPENAI_API_KEY=arcflare-dev-key qwen
```

## Model Support

Split any GGUF model across your cluster:

```bash
# Analyze a model and create split plan
gguf-splitter --model qwen2.5-32b-q4.gguf --layers-per-shard 10

# Load a model through the orchestrator
curl -X POST http://localhost:8000/api/models/load \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen2.5-32b", "path": "/models/qwen2.5-32b-q4.gguf"}'
```

## Target Hardware

| Device | Min RAM | Expected Performance |
|--------|---------|-------------------|
| ThinkPad X230 (i5, 8GB) | 8 GB | 1-3 layers of 32B |
| Dell Optiplex (i7, 16GB) | 16 GB | 3-8 layers of 32B |
| Raspberry Pi 4 (4GB) | 4 GB | 1-2 layers (slow) |
| Old phone (Linux/Android) | 4 GB | 1 layer (experimental) |
| Any x86 Linux machine | 2 GB | Can contribute |

## Project Structure

```
arcflare/
├── proto/                    # gRPC protocol definition
├── orchestrator/             # Python orchestrator
│   └── src/arcflare/
│       ├── api/              # OpenAI-compatible + management API
│       ├── cluster/          # Discovery, topology, partitioning
│       ├── models/           # Model download, sharding
│       ├── inference/        # Pipeline coordination
│       └── cli/              # OpenCode/Claude/Qwen adapters
├── node-agent/               # Rust node agent
│   └── src/
│       ├── hardware/         # CPU/GPU/RAM detection
│       ├── overclocking/     # Safe + aggressive tuning
│       ├── drivers/          # Driver auditing
│       ├── tuning/           # Kernel parameters
│       ├── inference/        # llama.cpp integration
│       └── network/          # gRPC + discovery
├── tools/
│   ├── gguf-splitter/        # GGUF sharding tool
│   └── scripts/              # Dev & deploy helpers
└── docker-compose.yml        # Multi-node dev env
```

## Roadmap

### Phase 1 ✅ (Current)
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

### Phase 2 🔜
- [ ] Full distributed inference pipeline
- [ ] P2P shard transfer (libp2p)
- [ ] Custom layer partitioning engine
- [ ] llama-cpp-4 inference integration
- [ ] KV cache optimization
### Phase 2 (maybes)
- [ ] Quantizing and training opportunity

### Phase 3 🔜
- [ ] Web dashboard
- [ ] Windows node agent support
- [ ] Crash recovery
- [ ] Live re-partitioning

## License

MIT

Built for the scrap hardware community. Give old devices a second life running AI.
