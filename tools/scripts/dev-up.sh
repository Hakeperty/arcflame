#!/usr/bin/env bash
set -euo pipefail

# ArcFlare dev environment setup
# Starts orchestrator + simulated nodes locally on one machine

ARCFLARE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
echo "Starting ArcFlare dev environment from: $ARCFLARE_DIR"

# 1. Start the orchestrator
echo "Starting orchestrator on :8000..."
cd "$ARCFLARE_DIR/orchestrator"
pip install -r requirements.txt -q
uvicorn arcflare.main:app --port 8000 --reload &
ORCH_PID=$!
echo "Orchestrator PID: $ORCH_PID"

# 2. Simulate nodes (only if the binary exists)
if command -v arcflare-node &> /dev/null; then
    for i in 0 1 2; do
        PORT=$((9001 + i))
        echo "Starting node-$i on :$PORT..."
        arcflare-node --grpc-port "$PORT" --name "node-$i" &
    done
else
    echo "Build node-agent first: cargo build -p node-agent"
fi

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║  ArcFlare dev environment running           ║"
echo "║  Orchestrator: http://localhost:8000         ║"
echo "║  API docs:     http://localhost:8000/docs    ║"
echo "╚══════════════════════════════════════════════╝"
echo ""
echo "Press Ctrl+C to stop"

trap "kill $ORCH_PID 2>/dev/null; exit 0" INT TERM
wait
