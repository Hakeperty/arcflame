#!/bin/sh

RPC_FLAGS=""
if [ "${ARCFLARE_ENABLE_RPC:-false}" = "true" ]; then
    RPC_FLAGS="--enable-rpc --rpc-port ${ARCFLARE_RPC_PORT:-10001} --rpc-server-bin ${ARCFLARE_RPC_BIN:-/usr/local/bin/llama-rpc-server}"
fi

# shellcheck disable=SC2086
exec /usr/local/bin/arcflare-node \
    --grpc-port "${ARCFLARE_GRPC_PORT:-9001}" \
    --name "${ARCFLARE_NODE_NAME:-node}" \
    --orchestrator-host "${ARCFLARE_ORCHESTRATOR_HOST:-orchestrator}" \
    --orchestrator-port "${ARCFLARE_ORCHESTRATOR_PORT:-8000}" \
    --log-level "${ARCFLARE_LOG_LEVEL:-info}" \
    $RPC_FLAGS
