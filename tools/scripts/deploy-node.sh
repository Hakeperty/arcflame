#!/usr/bin/env bash
set -euo pipefail

# Deploy the ArcFlare node agent to a remote machine via SSH

NODE="$1"
PORT="${2:-9001}"

if [ -z "$NODE" ]; then
    echo "Usage: $0 <user@host> [port]"
    exit 1
fi

echo "Deploying ArcFlare node to $NODE..."

# Build static binary
cd "$(dirname "$0")/../.."
cargo build --release -p node-agent
BINARY="target/release/node-agent"

# Deploy
scp "$BINARY" "$NODE:/tmp/arcflare-node"

# Install
ssh "$NODE" "sudo mv /tmp/arcflare-node /usr/local/bin/arcflare-node && sudo chmod +x /usr/local/bin/arcflare-node"

# Create systemd service
ssh "$NODE" "sudo tee /etc/systemd/system/arcflare-node.service > /dev/null" <<- SERVICE
[Unit]
Description=ArcFlare Cluster Node Agent
After=network.target

[Service]
ExecStart=/usr/local/bin/arcflare-node --grpc-port $PORT
Restart=always
RestartSec=5
User=nobody

[Install]
WantedBy=multi-user.target
SERVICE

ssh "$NODE" "sudo systemctl daemon-reload && sudo systemctl enable --now arcflare-node.service"

echo "✓ Node deployed and running!"
