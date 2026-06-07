.PHONY: proto build-node test-orchestrator dev clean

proto:
	@echo "Generating gRPC code..."
	cd proto && python -m grpc_tools.protoc \
		-I. \
		--python_out=../orchestrator/src/arcflare \
		--grpc_python_out=../orchestrator/src/arcflare \
		arcflare.proto
	cd proto && protoc \
		--rust_out=../node-agent/src \
		--grpc_out=../node-agent/src \
		--plugin=protoc-gen-grpc=$(which grpc_rust_plugin) \
		arcflare.proto || echo "Rust protoc not found, use cargo build instead"

build-node:
	cargo build --release -p node-agent

build-splitter:
	cargo build --release -p gguf-splitter

test-orchestrator:
	cd orchestrator && python -m pytest

dev:
	@echo "Starting dev environment..."
	@echo "Ensure you have built the node-agent first: make build-node"
	docker-compose up

clean:
	cargo clean
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf orchestrator/src/arcflare/*_pb2*.py
