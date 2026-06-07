#!/bin/sh
exec /usr/local/bin/arcflare-node \
    --grpc-port "${ARCFLARE_GRPC_PORT:-9001}" \
    --name "${ARCFLARE_NODE_NAME:-node}" \
    --orchestrator-host "${ARCFLARE_ORCHESTRATOR_HOST:-orchestrator}" \
    --orchestrator-port "${ARCFLARE_ORCHESTRATOR_PORT:-8000}" \
    --log-level "${ARCFLARE_LOG_LEVEL:-info}"
