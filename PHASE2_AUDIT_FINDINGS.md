# ArcFlare Audit — Confirmed Findings

Multi-agent review (49 agents), adversarially verified. 37 confirmed, 7 rejected.

## BUG (21)

### rpc-server child process is leaked and never reaped/killed when the agent exits
- **node-agent/src/main.rs:78-86** [rust-agent]
- On `--enable-rpc`, the agent spawns `llama-rpc-server` as a Child and then `Box::leak(Box::new(rpc))` the RpcServer. The comment calls the leak intentional, but the consequence is that RpcServer::stop() can never be called and there is no Drop/signal handler. tokio::process::Child is configured by default with kill_on_drop = false, and even if it were true, the value is leaked so Drop never runs. When the node agent dies (SIGTERM/SIGINT/normal exit), the rpc-server child keeps running orphaned, holding port grpc_port+1000. On agent restart, start() will then fail to bind because the old child still owns the port. There is no signal handling anywhere in the binary (no ctrl_c/SIGTERM hook) to trigger cleanup.
- **Fix:** The proposed fix is correct and complete: remove the Box::leak, keep RpcServer owned in main living until after serve() returns, install both tokio::signal::ctrl_c() and a unix SIGTERM handler (tokio::signal::unix::signal(SignalKind::terminate())) that call rpc.stop().await, and add `impl Drop for RpcServer` (or `.kill_on_drop(true)` on the Command) as a backstop. The finding rightly notes kill_on_drop(true) alone is insufficient while the value is leaked — both changes are needed. Concrete sketch: in main, after starting rpc, run `tokio::select!` between `network::grpc_server::serve(...)`, `ctrl_c()`, and the SIGTERM stream; on either signal call `rpc.stop().await` then return. Since rpc.process is `Arc<Mutex<Option<Child>>>`, you can clone the Arc into the signal task instead of moving the whole RpcServer. Minor note: `Child::kill().await` on Unix sends SIGKILL (not graceful) — fine for cleanup; if graceful shutdown is desired, send SIGTERM to the child first. Also recommend adding `impl Drop for RpcServer` that does a best-effort synchronous `start_kill()` for the panic/abnormal-exit path where async signal handlers don't run.

### Streaming handler re-initializes the global LlamaBackend on every request, which fails after the first
- **node-agent/src/network/grpc_server.rs:210-218** [rust-agent]
- forward_stream() calls `LlamaBackend::init()` directly inside spawn_blocking for each text_prompt request. llama.cpp's backend is a process-global singleton; the rest of the codebase correctly guards it with a `BACKEND` OnceCell in forward.rs (backend() at lines 16-25). Calling LlamaBackend::init() a second time (a second streaming request, or any concurrent forward()/load_shard() that already initialized BACKEND) returns an error (AlreadyInitialized) or is otherwise undefined, so the second and subsequent streaming requests fail with 'Backend init: ...'. It also bypasses the cached backend entirely.
- **Fix:** The proposed fix is correct and sound: replace the inline `LlamaBackend::init()` at grpc_server.rs:210 with the shared `crate::inference::forward::backend()` accessor and use the returned `&'static LlamaBackend` inside spawn_blocking. A `&'static LlamaBackend` is `Send + 'static`, so it can be moved into the closure. Concretely: resolve the backend before/at the start of the closure and surface its error as a stream error instead of constructing a fresh backend, e.g.:

    let be = match crate::inference::forward::backend() {
        Ok(b) => b,
        Err(e) => { let _ = tx.blocking_send(Err(Status::internal(e))); return; }
    };

(then use `be` for `LlamaModel::load_from_file(be, ...)` and `model.new_context(be, ...)`). Remove the now-unused `use llama_cpp_4::llama_backend::LlamaBackend;` import inside the closure.

Two completeness caveats worth flagging (not required to make the finding valid, but the fix as literally proposed leaves them): (a) The streaming path still reloads the GGUF from ARCFLARE_MODEL_PATH on every request rather than reusing the already-loaded shard model from get_loaded_model(); switching to the shared backend fixes the crash but not this redundant per-request model load. (b) Using the shared singleton makes concurrent streaming requests safe at the backend level, but they would still each create their own context/model load and contend on the GPU; that is a separate concern from this finding.

### argmax unwraps partial_cmp and panics on NaN logits
- **node-agent/src/network/grpc_server.rs:283-289** [rust-agent]
- The `argmax` closure does `a.partial_cmp(b).unwrap()`. If any logit is NaN (which can occur with bad model state, fp16/quantization edge cases, or a corrupt context), partial_cmp returns None and unwrap() panics inside spawn_blocking. The panic propagates as a JoinError, the tx sender is dropped, and the stream just terminates with an opaque broken-channel rather than a clean error — a denial-of-service vector driven by model/external input.
- **Fix:** The proposed fix is correct and sufficient to eliminate the panic. Prefer the idiomatic total-order comparator: replace `.max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())` with `.max_by(|(_, a), (_, b)| a.total_cmp(b))`. (`unwrap_or(Ordering::Equal)` also works but is less precise.) Note a residual nuance: total_cmp treats NaN as ordered (NaN sorts as the max for +NaN), so a NaN logit could be selected as argmax and produce a garbage token rather than panicking. That is strictly better than the DoS and consistent with the closure's existing best-effort `unwrap_or(0)` fallback. If correctness under NaN matters, additionally guard the buffer (e.g. break/return a clean Status::internal("non-finite logits") when `logits.iter().any(|x| x.is_nan())` before sampling) so a corrupt context yields a clean error instead of a junk token.

### EOS token is hardcoded to id 2 instead of the model's real EOS token
- **node-agent/src/network/grpc_server.rs:323** [rust-agent]
- Generation stops with `if sampled_id == 2 { break; }`. Token id 2 is only EOS for some tokenizers (e.g. Llama); Qwen2.5 (the default model in this very handler, qwen2.5-0.5b) uses a different EOS/EOT id. With the wrong EOS, generation never terminates early and always runs the full max_tokens, emitting garbage past the real end-of-text, or conversely stops mid-output if id 2 happens to be a valid content token.
- **Fix:** Replace the hardcoded `if sampled_id == 2 { break; }` with a model-driven end-of-generation check that covers ALL EOG tokens (EOS and EOT), not just a single token. In llama-cpp-4 0.3.x, `LlamaModel` exposes `is_eog_token(token: LlamaToken) -> bool` (wraps `llama_token_is_eog`), which is the correct predicate because Qwen2.5-instruct stops on `<|im_end|>` (the EOT token), which differs from the plain `token_eos()`. Compute it once before the loop is not possible (it depends on the sampled token), so check per-iteration:

    // line 323, replace `if sampled_id == 2 { break; }` with:
    if model.is_eog_token(last_token) { break; }

If `is_eog_token` is not available in this exact crate version, fall back to comparing against BOTH the model EOS and EOT tokens:

    let eos = model.token_eos();
    let eot = model.token_eot(); // if exposed; otherwise resolve "<|im_end|>" via str_to_token
    ...
    if last_token == eos || last_token == eot { break; }

Note `last_token` is already the `LlamaToken(sampled_id)` constructed at line 309, so prefer comparing the typed token rather than the raw i32. Do NOT hardcode 151643/151645 either, since the handler also honors an arbitrary `ARCFLARE_MODEL_PATH` model whose EOG ids may differ — always query the loaded model.

### Discovery shutdown flag is never set, and shutdown is checked only once per 5s interval
- **node-agent/src/main.rs:96-103** [rust-agent]
- main creates `let shutdown = Arc::new(RwLock::new(false))`, passes it by value into start_broadcaster (which moves it into the spawned task), and keeps no other reference. Nothing in the codebase ever writes `true` to it, so the broadcaster's `if *shutdown.read().await { break; }` is dead code and the broadcaster can never be stopped cleanly. Combined with the missing signal handling, the UDP broadcaster runs until hard process kill.
- **Fix:** The proposed fix is directionally correct but incomplete for a truly clean shutdown. (1) Retain a clone: change main.rs:96 to `let shutdown = Arc::new(RwLock::new(false));` and pass `shutdown.clone()` to start_broadcaster, keeping the original in main. (2) Install a signal handler that sets it, e.g. `tokio::spawn({ let s = shutdown.clone(); async move { let _ = tokio::signal::ctrl_c().await; *s.write().await = true; } });`. (3) Additionally — and this is the part the original fix omits — start_broadcaster currently returns () and drops the JoinHandle from tokio::spawn, so main cannot await the broadcaster's clean exit. Better solution: switch to a tokio::sync::watch (or Notify) channel and rewrite the loop to `select!` between the sleep and the shutdown signal so shutdown is observed immediately instead of up to 5s later, and have start_broadcaster return the JoinHandle so main can await it before returning. Example loop body: `tokio::select! { _ = rx.changed() => break, _ = tokio::time::sleep(DISCOVERY_INTERVAL) => { /* send */ } }`. Note also this is a graceful-shutdown/resource-cleanup issue, not a correctness/safety hazard during normal operation; severity bug is reasonable but on the lower end.

### Discovery uses a blocking std::net::UdpSocket inside an async tokio task
- **node-agent/src/network/discovery.rs:2, 57-69** [rust-agent]
- The broadcaster imports `std::net::UdpSocket` (blocking) and calls `socket.send_to(...)` inside a `tokio::spawn` async task. send_to on a std socket is a blocking syscall executed on a tokio worker thread; for broadcast it is usually fast, but if the socket buffer is full or the interface stalls it blocks the async runtime worker. The correct primitive is tokio::net::UdpSocket with .await.
- **Fix:** The proposed fix is correct and complete. Idiomatic option: replace `use std::net::UdpSocket` with `tokio::net::UdpSocket`. Since tokio's bind is async, move the bind into the spawned task (or await it before spawning) and await the send: `let socket = tokio::net::UdpSocket::bind("0.0.0.0:0").await?; socket.set_broadcast(true).ok(); ... if let Err(e) = socket.send_to(&payload, format!("255.255.255.255:{}", DISCOVERY_PORT)).await { warn!(...); }`. Note set_broadcast on tokio's socket is still synchronous and returns Result, so `.ok()` is fine. Alternatively keep the std socket and wrap the send in `tokio::task::spawn_blocking`. Either resolves the blocking-call-on-runtime-worker issue.

### set_broadcast error is silently swallowed; broadcast sends will then fail every interval
- **node-agent/src/network/discovery.rs:29** [rust-agent]
- `s.set_broadcast(true).ok();` discards the Result. If enabling SO_BROADCAST fails, the socket proceeds without broadcast permission and every subsequent send_to to 255.255.255.255 fails with EACCES, producing a warn log every 5 seconds forever while discovery silently never works. The root failure is hidden.
- **Fix:** The proposed fix is correct. Concretely, since start_broadcaster returns () and cannot propagate, handle the error at the bind site and return early before spawning the loop. Replace line 29's `s.set_broadcast(true).ok();` with: `if let Err(e) = s.set_broadcast(true) { warn!("Failed to enable SO_BROADCAST on discovery socket: {}", e); return None; } else { Some(s) }` — restructuring the match arm to surface the failure once and skip starting the broadcaster. Minimal form within the existing Ok arm: `Ok(s) => { if let Err(e) = s.set_broadcast(true) { warn!("Failed to enable broadcast on discovery socket: {}", e); return; } s }`. This surfaces the root failure exactly once instead of as a recurring send error every 5 seconds.

### rpc_port = grpc_port + 1000 can panic on overflow in debug builds
- **node-agent/src/main.rs:72** [rust-agent]
- `args.rpc_port.unwrap_or(args.grpc_port + 1000)` adds 1000 to a u16. With grpc_port > 64535 (valid u16 values up to 65535, settable via CLI) this overflows u16, which panics in debug builds and wraps to a tiny/privileged port in release builds. External CLI input directly controls grpc_port.
- **Fix:** The proposed fix is correct in spirit but should also reject the disabled-branch edge cases and a 0/privileged result. Concretely, replace line 71-75 with checked arithmetic that errors cleanly:

```rust
let rpc_port: u16 = if args.enable_rpc {
    match args.rpc_port {
        Some(p) => p,
        None => args.grpc_port.checked_add(1000).ok_or_else(|| {
            format!(
                "grpc_port {} + 1000 overflows u16; pass --rpc-port explicitly",
                args.grpc_port
            )
        })?,
    }
} else {
    0
};
```

This returns the `Box<dyn Error>` from `main` (the `?` works since `String: Into<Box<dyn Error + Send + Sync>>`) instead of panicking or silently wrapping. Optionally also validate the result is a non-privileged port (>= 1024) when derived, as the original finding suggests, since even a non-overflowing default could land below 1024 only via wrap — but with checked_add that cannot happen, so the privileged-port concern is fully resolved by the checked add alone.

### _find_model ignores model_name and returns the first .gguf found (wrong model, non-deterministic)
- **orchestrator/src/arcflare/inference/pipeline.py:29-40** [py-pipeline]
- _find_model(model_name) never uses model_name. It iterates os.listdir(models_dir) and returns the first entry ending in .gguf. With more than one model present, the wrong model is selected, and os.listdir order is arbitrary (filesystem-dependent, not sorted), so the selection is non-deterministic across runs/machines. The caller always passes the requested model name (run/run_stream/_rpc_distributed_inference) but it is silently discarded. _local_inference even hardcodes _find_model('default'). Also os.listdir raises FileNotFoundError if models_dir does not exist, which is uncaught in _rpc_distributed_inference / _distributed_inference paths (only _local_inference catches generic Exception).
- **Fix:** In _find_model, actually use model_name, sort for determinism, and guard the directory. Example:

def _find_model(self, model_name: str) -> Optional[str]:
    models_dir = os.environ.get(
        "ARCFLARE_MODELS_DIR",
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "models"),
    )
    models_dir = os.path.abspath(models_dir)
    if not os.path.isdir(models_dir):
        return None
    ggufs = sorted(f for f in os.listdir(models_dir) if f.endswith(".gguf"))
    if not ggufs:
        return None
    # strip namespace ("arcflare/foo" -> "foo") and extension for matching
    wanted = (model_name or "").split("/")[-1].removesuffix(".gguf").lower()
    if wanted:
        for fname in ggufs:
            if wanted in fname.lower():
                return os.path.join(models_dir, fname)
    # deterministic single-file / first-match fallback only when name does not resolve
    return os.path.join(models_dir, ggufs[0])

Additionally (beyond the finding's proposed fix, to fully close the defect): _local_inference should not hardcode "default". Thread the real model name through — change `async def _local_inference(self, prompt, ...)` to accept a `model: str` parameter and call `self._find_model(model)`, and update all call sites (run_stream line 129, _distributed_inference lines 280/319 path, and _grpc_inference fallback line 319) to pass the model argument. Without this, even after fixing _find_model, the local fallback path still ignores the requested model.

### gRPC mode (Mode 2) is dead on first request: active_connections computed before _connect_to_nodes
- **orchestrator/src/arcflare/inference/pipeline.py:113-118** [py-pipeline]
- In run_stream, active_connections = list(self.node_connections.keys()) is read at line 114 BEFORE any connection is established. _connect_to_nodes is only called at line 118, inside the if-body. On a fresh InferencePipeline, node_connections is empty, so active_connections is empty, so the guard `if alive and active_connections` is False and Mode 2 (gRPC streaming) is skipped entirely, jumping straight to local fallback. The gRPC path can therefore never run on the first inference request; it only becomes reachable on a later request after connections happen to have been populated by some other code path. The connect-then-check ordering is inverted.
- **Fix:** The proposed fix is correct in direction. Replace lines 113-118 so the connect happens before the gate, and gate on the resulting connection set rather than the pre-connect snapshot:

    alive = [n for n in nodes if n.get("grpc_port") and n.get("ip_address")]
    if alive:
        await self._connect_to_nodes(alive)
        if self.node_connections:
            logger.info(f"gRPC stream mode: {len(alive)} nodes available")
            try:
                async for token in self._distributed_inference(
                    model, prompt, max_tokens, temperature, alive
                ):
                    yield token
                return
            except Exception as e:
                logger.warning(f"gRPC inference failed ({e}), falling back to local")

Note: _connect_to_nodes can raise (client.connect() at line 71), so it should be inside the try/except (or otherwise guarded) so a connection failure still falls through to local rather than propagating out of run_stream. The existing await self._connect_to_nodes(nodes) at line 270 inside _distributed_inference then becomes redundant; it is harmless (line 66 skips already-connected node_ids) but can be removed for clarity.

### Subprocess not killed on timeout — orphaned llama-cli process / resource leak
- **orchestrator/src/arcflare/inference/pipeline.py:180-209** [py-pipeline]
- In _rpc_distributed_inference, asyncio.wait_for(proc.communicate(), timeout=600) raises TimeoutError after 600s, but the spawned llama-cli process is never terminated. wait_for cancels the communicate() coroutine but does not kill the child. The llama-cli subprocess (which can be pinning GPU/CPU and the whole RPC cluster) keeps running orphaned, and its PIPE-buffered output is leaked. The same defect exists in _local_inference at lines 371-373/397-399. Repeated timeouts accumulate orphaned processes.
- **Fix:** The proposed fix is correct in substance; harden it slightly to handle already-exited processes and a hung kill. Apply the same pattern to BOTH call sites. Wrap the spawn-and-communicate in try/finally so the process is reaped on TimeoutError AND on async-generator cancellation (client disconnect). Example for _rpc_distributed_inference:

    proc = None
    try:
        proc = await asyncio.create_subprocess_exec(*cmd, stdout=PIPE, stderr=PIPE)
        stdout_data, stderr_data = await asyncio.wait_for(proc.communicate(), timeout=600)
        ... (existing parsing / yields) ...
    except asyncio.TimeoutError:
        logger.error("llama-cli RPC inference timed out")
        raise RuntimeError("RPC inference timed out")
    finally:
        if proc is not None and proc.returncode is None:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                pass  # could not reap within 5s; nothing more to do

For _local_inference, keep its existing behavior of yielding a message instead of raising on TimeoutError, but add the same `finally` block that kills+reaps proc when proc.returncode is None. Note: yielding inside an `except`/before the `finally` in an async generator is fine, but the kill must live in `finally` so it also runs when the generator is closed (GeneratorExit) — catching only asyncio.TimeoutError would miss the client-disconnect/cancellation path that openai.py's SSE streaming can trigger. Guarding with `proc.returncode is None` avoids killing a process that already exited normally, and ProcessLookupError must be swallowed for the race where it died between the check and kill.

### Async generators are abandoned without cleanup, leaving subprocesses running if the consumer stops early
- **orchestrator/src/arcflare/inference/pipeline.py:174-205** [py-pipeline]
- _rpc_distributed_inference and _local_inference spawn a subprocess and then yield tokens in a loop with await asyncio.sleep(...). If the downstream consumer (e.g. an HTTP/SSE client that disconnects) stops iterating the async generator, the generator is suspended at a yield and GeneratorExit may be thrown in, but there is no try/finally to terminate proc. The llama-cli process keeps running to completion (up to 600s) with no reader, leaking the process and its pipes. forward_stream in grpc_client.py has the same shape: if the caller breaks out of the async-for early, the underlying gRPC stream call is not cancelled/closed.
- **Fix:** Wrap the subprocess lifecycle in try/finally so the child is killed and reaped on ANY exit path (normal return, exception, or GeneratorExit from consumer disconnect). The proc must be created OUTSIDE the awaited communicate() so the finally can see it. Example for _rpc_distributed_inference / _local_inference:

    proc = await asyncio.create_subprocess_exec(*cmd, stdout=PIPE, stderr=PIPE)
    try:
        stdout_data, stderr_data = await asyncio.wait_for(proc.communicate(), timeout=600)
        ... parse/yield loop ...
    finally:
        if proc.returncode is None:
            proc.kill()
            try:
                await proc.wait()
            except Exception:
                pass

Note: because communicate() runs to completion before yielding, the only window where the child is actually alive is the wait_for(communicate()) await; the finally above covers cancellation there. (Streaming token-by-token while the process runs would require reading proc.stdout incrementally instead of communicate(); that is a larger change and not required to fix the leak.)

For grpc_client.forward_stream, capture the streaming call object and cancel it in finally:

    if not self._stub:
        return
    call = self._stub.ForwardStream(iter([req]), timeout=300)
    try:
        async for resp in call:
            yield resp
            if not resp.has_logits:
                break
    except Exception as e:
        logger.error(f"ForwardStream on {self.node_id} failed: {e}")
        raise
    finally:
        call.cancel()   # idempotent; cancels the underlying HTTP/2 stream on early break/GeneratorExit

(grpc.aio.UnaryStreamCall / StreamStreamCall.cancel() is safe to call after normal completion.)

### Banner-stripping logic can drop real model output and break valid lines
- **orchestrator/src/arcflare/inference/pipeline.py:197-204, 385-394** [py-pipeline]
- The output parser makes several content-dropping assumptions: (1) capturing only starts after a line that .strip() == '> ' / startswith('> '); if llama-cli's first generated token shares the prompt line or the '>' marker is absent (varies by build/flags), nothing is ever captured and the whole response is silently dropped. (2) `if capturing and stripped.startswith('[')` breaks the loop — any legitimate generated line that begins with '[' (markdown, code, citations, JSON arrays) terminates output prematurely. (3) In _local_inference, `if capturing and stripped.startswith('/')` skips the line entirely — real output lines starting with '/' (paths, fractions, regexes) are silently dropped. (4) lstrip('|/-\\=') strips leading characters from genuine content that happens to start with those chars. (5) `stripped.lstrip('\b \b')` at line 391 is a no-op confusion: lstrip takes a char set, so it just strips backspace/space, not the intended spinner sequence. The empty-line `continue` at 192/381 also collapses blank lines in the real output.
- **Fix:** The proposed fix is correct in direction but should account for what the code already does. Both invocations ALREADY pass `--no-display-prompt` (lines 164, 360), so stdout already contains only generated text plus possible spinner/control bytes — there is no need for the `> ` gate at all. Concretely:

1. Remove the `> `-marker gating entirely (delete the `capturing` state machine). Treat all stdout lines as generated content. If a leading banner must be skipped, detect it explicitly (e.g., skip lines until the first non-empty line, or match known llama.cpp banner prefixes like `build:`, `main:`, `llama_`, `system_info:`), rather than waiting for a `>` that --no-display-prompt prevents from appearing.

2. Do NOT `break` on `[` or `Exiting` as content tests. llama.cpp end-of-generation markers appear on stderr or as bracketed control tokens like `[end of text]`; match that exact token (`stripped == "[end of text]"`) rather than any line starting with `[`. Keep `Exiting` only if it is matched as a standalone llama.cpp shutdown line, not a substring of content.

3. Delete the `startswith("/") : continue` rule (line 387) — it has no legitimate purpose and drops real content.

4. Do not lstrip spinner chars from content. Spinner animation in llama.cpp is emitted via carriage-return/backspace overwrite on the SAME stream; strip only `\r` and `\b` control bytes (`.replace("\r", "").replace("\b", "")`) and leave `|/-\=` alone. If a spinner prefix must be removed, only remove a single leading spinner glyph when it is immediately followed by more control/whitespace (i.e., it is clearly an animation frame, not content).

5. Do not collapse blank lines: preserve paragraph structure by yielding empty lines too, or at least preserve a single blank line between paragraphs instead of `continue`-ing on every empty line.

6. Strongly preferred over all heuristics: switch to the llama.cpp server (`llama-server`) JSON/`/completion` endpoint, or parse the structured JSONL that `llama-cli` can emit, so output is unambiguous and no line-scanning is needed. The proposed fix is correct that this is the robust path; the heuristic patches above are the fallback if the server mode is not adopted.

### _load_shards_on_nodes uses nodes.index(node) for ordering/positions, which is O(n^2) and breaks on duplicate node dicts
- **orchestrator/src/arcflare/inference/pipeline.py:225-227** [py-pipeline]
- Inside the for-loop over nodes, the code calls nodes.index(node) three times to compute first layer, num layers, and has_head. list.index returns the index of the FIRST equal element, so if two node dicts compare equal (same content) the later node gets the earlier node's position, producing overlapping/duplicate layer ranges and the wrong has_lm_head assignment (two heads or none). It is also O(n^2). The boundary math (first = index*layers_per) can also leave a gap or overlap because layers_per = total//n and only the last node gets the remainder via the index check, which is itself index-based.
- **Fix:** Replace the per-element index() calls with a stable enumerate index. Inside `_load_shards_on_nodes`, change the loop header to capture the position and derive all three values from it (also hoist the constants that do not change per node):

    total_layers = self._get_model_layer_count(model_path) or 24
    n_nodes = max(1, len(nodes))
    layers_per = total_layers // n_nodes
    for i, node in enumerate(nodes):
        node_id = node.get("node_id", "")
        client = self.node_connections.get(node_id)
        if not client:
            continue
        first = i * layers_per
        is_last = (i == n_nodes - 1)
        num = (total_layers - first) if is_last else layers_per
        has_head = is_last
        status = await client.load_shard(
            model_name="arcflare/default",
            gguf_path=model_path,
            first_layer=first,
            num_layers=num,
            has_lm_head=has_head,
        )
        if status and status.loaded:
            loaded += 1

This makes positions stable regardless of duplicate node contents, removes the O(n^2) repeated scans, and preserves the original behavior of giving the remainder layers to the final shard. Note: skipped (unconnected) nodes still consume their layer slot via the enumerate index, which matches the original code's intent of slicing by node ordinal; if instead only connected nodes should receive shards, filter nodes to connected ones before enumerating.

### Bare `return` inside async generators yields nothing — silent empty result with no fallback
- **orchestrator/src/arcflare/inference/pipeline.py:265-268, 290-293, 296-299** [py-pipeline]
- _distributed_inference returns early (line 268) when no model is found; _grpc_inference returns early (lines 292-293, 298-299) when there is no target/client. In an async generator, `return` just ends iteration, yielding zero tokens. Because these are awaited within run_stream's `try: async for token in ...: yield token; return` and they complete WITHOUT raising, run_stream hits the `return` at line 123 and the caller receives an empty string with no error and no fallback to local. The user gets a silent empty completion instead of a usable result.
- **Fix:** The proposed fix's intent is correct, but prefer falling through to `_local_inference` over raising, to stay consistent with this module's existing degradation pattern (see lines 279-281 where `_distributed_inference` already falls back to local on shard-load failure, and lines 316-320 where `_grpc_inference` already falls back to local on streaming exceptions). Concretely:

In `_distributed_inference` (lines 265-268), replace the bare `return` with a local fallback:
    model_path = self._find_model(model)
    if not model_path:
        logger.warning("No model found for distributed inference, using local")
        async for token in self._local_inference(prompt, max_tokens, temperature):
            yield token
        return

In `_grpc_inference` (lines 290-299), replace each bare `return` similarly so both the `not target` and `not client` branches fall back:
    target = self._pick_target_node(nodes)
    if not target:
        logger.warning("No available node for inference, using local")
        async for token in self._local_inference(prompt, max_tokens, temperature):
            yield token
        return
    node_id = target.get("node_id", "")
    client = self.node_connections.get(node_id)
    if not client:
        logger.warning(f"Node {node_id} not connected, using local")
        async for token in self._local_inference(prompt, max_tokens, temperature):
            yield token
        return

Raising (so run_stream's `except Exception` falls through to mode 3) is a valid alternative and also fixes the bug, but the explicit local fallback matches the surrounding code and avoids relying on run_stream's catch-all. Note that `_local_inference` itself never yields empty — it emits a stub message when no model/binary is present (lines 342-352) — so falling back there guarantees a non-empty result.

### Streaming SSE emits Python dict repr instead of JSON, breaking all OpenAI clients
- **/home/harty/projekt/orchestrator/src/arcflare/api/openai.py:143-153, 168-177** [py-api]
- generate_chat_stream and generate_completion_stream yield dicts whose 'data' key is itself a dict ({"choices": [...]}). sse_starlette's ensure_bytes() (site-packages/sse_starlette/event.py:93-95) treats a yielded dict as ServerSentEvent constructor kwargs: ServerSentEvent(**data). The inner dict is stored as self.data, and encode() (event.py:50) writes `data: {str(self.data)}`. str() on a dict produces a Python repr with single quotes (e.g. {'choices': [{'index': 0, ...}]}), which is NOT valid JSON. Every OpenAI-compatible client (openai SDK, LangChain, etc.) does json.loads on the SSE data line and will fail. The non-streaming path works because it returns a normal dict via FastAPI's JSON encoder, but the streaming path is completely broken. Additionally the chunk shape is missing required OpenAI fields: 'object': 'chat.completion.chunk', 'created', 'model', and the first chunk should include delta.role='assistant'.
- **Fix:** The proposed fix is correct in mechanism; refine it slightly. Add `import json` (only `time`, `uuid`, `logging` are imported now). Generate ONE stable id per stream (not a fresh uuid per chunk) and reuse it. For generate_chat_stream:

```python
async def generate_chat_stream(request):
    from ..inference.pipeline import run_inference_stream
    cmpl_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    first = True
    async for chunk in run_inference_stream(model=request.model, prompt=format_messages(request.messages), max_tokens=request.max_tokens, temperature=request.temperature):
        delta = {"content": chunk}
        if first:
            delta = {"role": "assistant", "content": chunk}
            first = False
        yield {"data": json.dumps({"id": cmpl_id, "object": "chat.completion.chunk", "created": created, "model": request.model, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]})}
    # final chunk with finish_reason then DONE sentinel
    yield {"data": json.dumps({"id": cmpl_id, "object": "chat.completion.chunk", "created": created, "model": request.model, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})}
    yield {"data": "[DONE]"}
```

For generate_completion_stream use object="text_completion" and a "text" field (not "delta"), e.g. choices=[{"index": 0, "text": chunk, "finish_reason": None}], a final chunk with finish_reason="stop", then yield {"data": "[DONE]"}. Drop the custom event:"delta"/event:"done" names as the finding states (yielding only the "data" key avoids emitting an event: line). Note: yielding {"data": "<json str>"} works because ensure_bytes constructs ServerSentEvent(data="<json str>", sep=...) and encode() does str() on a string (no-op), producing a valid `data:` line.

### cluster_status reads node fields that do not exist in serialized NodeInfo (always zero)
- **/home/harty/projekt/orchestrator/src/arcflare/api/management.py:82-83** [py-api]
- discovery_service.get_nodes() returns asdict(NodeInfo) where the only structured hardware field is 'hardware' (discovery.py:24). cluster_status does n.get('memory', {}).get('total_bytes', 0) and n.get('gpus'), but NodeInfo has no 'memory' or 'gpus' keys — those live (if anywhere) under node.hardware. As a result total_ram_gb is always 0 and total_gpus is always 0 regardless of actual cluster hardware. The same mismatch exists in cluster/partition.py _score_nodes (lines 116-135: memory, gpus, benchmark_score, latency_ms), so every node always scores the OS-reserve fallback and partitioning is effectively uniform/meaningless.
- **Fix:** The proposed fix is correct in shape but must guard against hardware being None (it is Optional[dict] and currently always None). Two parts:

1) Read-path indirection (immediate correctness for the key mismatch):
management.py cluster_status:
    hw_list = [(n.get("hardware") or {}) for n in nodes]
    total_ram = sum(hw.get("memory", {}).get("total_bytes", 0) for hw in hw_list)
    total_gpus = sum(1 for hw in hw_list if hw.get("gpus"))
partition.py _score_nodes (per node):
    hw = node.get("hardware") or {}
    memory = hw.get("memory", {})
    total_ram = memory.get("total_bytes", 8 * 1024**3)
    available_ram = memory.get("available_bytes", total_ram)
    gpus = hw.get("gpus", [])
    benchmark = hw.get("benchmark_score", node.get("benchmark_score", 0))  # benchmark_score is a top-level field on HardwareReport, so it lives under hardware
    # latency_ms is NOT on HardwareReport; it lives on hardware.network.links[].latency_ms (per proto), so node.get('latency_ms') will never resolve. Either drop the latency penalty or read it from hardware['network']['links'].

2) REQUIRED root cause (without this, step 1 still returns zeros/defaults): nothing ever populates NodeInfo.hardware. A gRPC GetHardwareInfo/ReportNodeStatus -> HardwareReport ingestion path must store the report dict into self.nodes[node_id].hardware (or flatten its fields onto NodeInfo) at registration/discovery time. Until that ingestion exists, total_ram_gb/total_gpus remain 0 and all partition scores remain the uniform OS-reserve fallback even with the indirection applied.

### UDP-discovered nodes are never pruned because last_seen stays 0.0
- **/home/harty/projekt/orchestrator/src/arcflare/cluster/discovery.py:85-93, 130-131** [py-api]
- On first discovery, NodeInfo is created without setting last_seen, so it defaults to 0.0 (dataclass default at line 23). _prune_dead_nodes only considers nodes where n.last_seen > 0 (line 131). Therefore a node that broadcasts a single UDP packet and then dies is never pruned — it is permanently immune to the heartbeat timeout. last_seen is only set on the SECOND and later packets (line 95). A node must be seen at least twice before the timeout can ever apply to it.
- **Fix:** The proposed fix is correct: add `last_seen=time.time()` to the first-discovery NodeInfo(...) constructor at discovery.py:85-93 (time is already imported at line 5). For full consistency with both the else-branch (line 96) and the HTTP register path (management.py:48), also set `status="alive"` in that constructor instead of leaving the dataclass default of "discovered" — though that status change is a nit and not strictly required to fix the pruning bug. The load-bearing change is the last_seen assignment.

### register_node performs no validation and overwrites existing nodes unconditionally
- **/home/harty/projekt/orchestrator/src/arcflare/api/management.py:38-50** [py-api]
- register_node trusts req fully: no validation that node_id is non-empty, that grpc_port/rpc_port are in 1..65535, or that name is present. An empty node_id is accepted and stored under key '' (and an empty-id UDP packet at discovery.py:75 likewise becomes a phantom node). client_ip is taken from request.client.host, which behind a proxy/load-balancer is the proxy IP, not the node — so get_rpc_endpoints will emit unreachable host:port. There is also no auth on this endpoint, so any client on the network can inject/overwrite cluster nodes (combined with CORS allow_origins=['*'] in main.py:43).
- **Fix:** Use Pydantic v2 idioms (project pins pydantic>=2.10.0). In management.py: from typing import Annotated; from pydantic import BaseModel, Field. Then:

class RegisterRequest(BaseModel):
    node_id: Annotated[str, Field(min_length=1, max_length=128)]
    name: Annotated[str, Field(min_length=1, max_length=256)]
    grpc_port: Annotated[int, Field(ge=1, le=65535)] = 9001
    rpc_port: Annotated[int, Field(ge=0, le=65535)] = 0   # 0 = no rpc-server (sentinel), so allow 0 — NOT ge=1
    version: str = "0.0.0"
    os: str = "unknown"

In cluster/discovery.py:_handle_discovery, reject empty/missing node_id and out-of-range ports before inserting:
    node_id = msg.get("node_id", "")
    if not node_id:
        logger.debug(f"Discovery msg from {addr} missing node_id; ignored")
        return
    grpc_port = msg.get("grpc_port", 0); rpc_port = msg.get("rpc_port", 0)
    if not (0 <= grpc_port <= 65535) or not (0 <= rpc_port <= 65535):
        return

Address trust/source-IP: prefer an explicit advertised address from the node body over the connection peer. Add an optional advertise_host to RegisterRequest and use it when present, else fall back to request.client.host. Only honor X-Forwarded-For if a trusted-proxy setting is enabled (blindly trusting XFF lets any client spoof the source IP, which is worse than request.client.host). Document that for the UDP path the peer addr[0] is the only available source.

Auth: protect the management router with a shared-secret token, e.g. a FastAPI dependency that checks an Authorization/X-ArcFlare-Token header against a configured value, applied to register_node (and ideally the cluster/tune,benchmark mutating endpoints). 

CORS (main.py:43-44): allow_origins=['*'] together with allow_credentials=True is itself invalid — browsers reject credentialed wildcard responses. Restrict allow_origins to a configured allowlist, or set allow_credentials=False if no credentialed cross-origin access is needed.

### Dockerfile.orchestrator COPYs models/qwen2.5-0.5b-instruct-q4_k_m.gguf which does not exist (only tinyllama is present) — build fails
- **/home/harty/projekt/Dockerfile.orchestrator:18** [deploy-proto]
- Line 18 `COPY models/qwen2.5-0.5b-instruct-q4_k_m.gguf /models/` references a file that is absent: the only model in `models/` is `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf`. `docker build -f Dockerfile.orchestrator` fails with `COPY failed: ... qwen2.5-0.5b-instruct-q4_k_m.gguf: no such file or directory`. Compounding nuance: the docker-compose service also mounts `./models:/models`, which would override whatever was COPY'd anyway, and the orchestrator's `_find_model()` just picks the first *.gguf in the dir — so the COPY is both broken and redundant. The same qwen path is hardcoded as the node's default model in grpc_server.rs (line ~222), so even the running node will look for a model that isn't present.
- **Fix:** Drop the model COPY and rely on the volume mount: replace lines 16-18 with just `RUN mkdir -p /models` (the docker-compose `./models:/models` mount and pipeline.py _find_model's first-*.gguf scan make any baked-in model both broken and redundant). For the node default in grpc_server.rs:222, prefer making it required (return Status::internal if ARCFLARE_MODEL_PATH is unset) since node-alpha/node-beta also only have the volume mount and no guaranteed filename; if a fallback is kept, it must use the EXACT existing filename with correct casing: "/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf" (uppercase Q4_K_M). Note pipeline.py:37 uses case-sensitive `.endswith(".gguf")` (lowercase) which happens to match the present file's lowercase `.gguf` extension, so no change needed there.

### docker-compose mounts ./orchestrator:/app, shadowing the in-image code layout so `uvicorn arcflare.main:app` fails with ModuleNotFoundError
- **/home/harty/projekt/docker-compose.yml:13** [deploy-proto]
- Dockerfile.orchestrator copies `orchestrator/src/` into `/app` (line 10), so in the image `/app/arcflare/main.py` exists and `uvicorn arcflare.main:app` resolves (PYTHONPATH=/app). But docker-compose line 13 mounts `./orchestrator:/app`, which at runtime replaces `/app` with the host `orchestrator/` directory. After the mount, `/app` contains `requirements.txt` and `src/` — there is NO `/app/arcflare/`, the real package is at `/app/src/arcflare/`. So `uvicorn arcflare.main:app` raises `ModuleNotFoundError: No module named 'arcflare'` and the orchestrator container fails to start under compose (only the bare image would work). The mount path is off by the `src/` segment.
- **Fix:** The proposed fix is correct and complete: change docker-compose.yml line 13 from `- ./orchestrator:/app` to `- ./orchestrator/src:/app`. This makes the host's src/ (containing arcflare/) the /app directory, so /app/arcflare/main.py exists and `arcflare.main:app` resolves under PYTHONPATH=/app, matching the image layout from `COPY orchestrator/src/ /app`. (Minor, non-blocking aside: orchestrator/src/arcflare/ contains __pycache__ .pyc files compiled for cpython-313 while the image is python:3.11; 3.11 ignores them via magic-number/mtime cache invalidation, so they do not affect this fix.)

## PERF (9)

### Streaming handler reloads the entire model from disk on every request instead of using the loaded shard
- **node-agent/src/network/grpc_server.rs:220-236** [rust-agent]
- For each text_prompt streaming request, the handler reads ARCFLARE_MODEL_PATH and calls LlamaModel::load_from_file inside spawn_blocking, ignoring the shard already loaded via load_shard() (model::CURRENT_SHARD). Loading a GGUF from disk per request adds large latency and memory churn, and diverges from the rest of the pipeline which expects load_shard/forward to operate on the pre-loaded model. It also hardcodes a default path unrelated to the configured shard.
- **Fix:** In forward_stream's text_prompt branch, drop the per-request model load and direct backend init. Acquire the singleton backend via crate::inference::forward::backend() and acquire the preloaded model via crate::inference::model::get_loaded_model(); error out with Status::failed_precondition if no shard is loaded. Keep the existing autoregressive generation loop (context creation, prompt decode, argmax sampling, streaming token sends) but run it against the borrowed shard.model. Because LlamaModel is not 'static and the read guard must be held across the synchronous llama-cpp calls, you cannot trivially move it into spawn_blocking; either (a) clone the Arc<RwLock<Option<ShardState>>> and take the read lock inside the blocking task (using a blocking read or by passing an owned guard), or (b) restructure so the inference runs while holding guard.read().await as run_forward_pass does (which avoids spawn_blocking and instead does the synchronous work inline, since there are no awaits during the llama calls). Either way, the model is loaded once at load_shard time and reused across all streaming requests, and the hardcoded ARCFLARE_MODEL_PATH/default GGUF path is removed entirely so generation operates on the configured shard.

### gRPC connections are never closed — channel/FD leak
- **orchestrator/src/arcflare/inference/pipeline.py:62-74** [py-pipeline]
- _connect_to_nodes creates NodeGrpcClient objects (each opening a grpc.aio.insecure_channel) and stores them in self.node_connections, keyed by node_id, for the lifetime of the singleton pipeline. NodeGrpcClient.close() exists in grpc_client.py (line 150) but is never called anywhere in the pipeline. When a node is pruned/dies (discovery._prune_dead_nodes removes it), its entry in node_connections persists with a live-but-dead channel, and a new connection is opened if it reappears. Over time this leaks aio channels and their sockets/FDs, and stale connections are picked by _pick_target_node (random.choice over connected) causing failed inferences.
- **Fix:** The fix direction is right but incomplete. Add an async reconciliation step that runs at the START of each run_stream (before _connect_to_nodes) so stale entries are removed first — otherwise the `node_id not in self.node_connections` guard at pipeline.py:66 blocks reconnecting a node that dropped and reappeared, leaving it stuck on a dead channel.

Concretely, add a helper and call it from run_stream right after fetching `nodes`:

    async def _reconcile_connections(self, nodes: list[dict]):
        live_ids = {n.get("node_id", "") for n in nodes if n.get("node_id")}
        for node_id in list(self.node_connections.keys()):
            if node_id not in live_ids:
                client = self.node_connections.pop(node_id)
                try:
                    await client.close()
                except Exception as e:
                    logger.warning(f"Error closing gRPC client for {node_id}: {e}")
                logger.info(f"Closed stale gRPC connection to node {node_id}")

In run_stream, after `nodes = discovery_service.get_nodes()...`:
    await self._reconcile_connections(nodes)

Additional notes the original fix misses:
1. Also close connections on full pipeline shutdown — add an `async def aclose(self)` that closes all clients, and wire it into the FastAPI/app shutdown in main.py (currently main.py has no pipeline cleanup at all).
2. The original fix says 'when a connection check fails' — there is no active health check; the practical signal is membership in discovery's node set (reconcile against it) plus catching forward_stream failures. On a streaming failure in _grpc_inference (line 316), consider closing+removing that client so the next run re-establishes it, rather than reusing a possibly-broken channel.
3. Note `nodes` is rebuilt via asdict() each call (discovery.py:105), so `nodes.index(node)` in _load_shards_on_nodes works on a fresh list — reconciliation by node_id (not object identity) is the correct key, as written above.

### Fresh llama-cli subprocess per request reloads the entire model from disk every time
- **/home/harty/projekt/orchestrator/src/arcflare/inference/pipeline.py:174-182 (RPC), 365-373 (local)** [perf]
- Both _rpc_distributed_inference and _local_inference do asyncio.create_subprocess_exec([llama_cli, '-m', model_path, ...]) on every single request. llama-cli loads the full GGUF model into RAM/VRAM, builds the compute graph, (for RPC) reconnects to every rpc-server and re-shards the model across the cluster, runs one prompt, then exits and frees everything. For a multi-GB model this is dominated by model-load time (seconds to tens of seconds of cold load + mmap fault-in) on every call, so per-request latency is essentially constant regardless of prompt size and throughput collapses under concurrency. There is no persistent worker, no warm process, and no reuse of the loaded weights between requests.
- **Fix:** The proposed fix is correct and recommended: run a single long-lived llama-server (llama.cpp's HTTP server) started once at orchestrator startup (or lazily on first request, guarded by an asyncio.Lock to avoid concurrent double-spawn) with -m <model> --rpc <endpoints> for the distributed path, keep it resident, and issue per-request calls to its /completion endpoint (use stream:true SSE for true token streaming). Manage one server per distinct model/RPC-topology and tear it down on shutdown. This loads the model and RPC tensor-split exactly once; subsequent requests pay only prompt-eval + generation and can share KV/prompt caching. Additional note specific to this code: the current stdout parsing keys off a '> ' interactive-REPL prompt marker (lines 194 and 382) and strips spinner glyphs, which is brittle interactive-output framing inconsistent with the non-interactive --single-turn mode actually requested; migrating to llama-server's structured JSON/SSE responses also eliminates this fragile text scraping. If llama-server is unavailable, the lower-effort interim alternative is a single persistent interactive llama-cli process fed via stdin, but that retains the brittle stdout parsing and lacks proper concurrency, so llama-server is the preferred target.

### _find_llama_cli does filesystem stat/access probing on every request
- **/home/harty/projekt/orchestrator/src/arcflare/inference/pipeline.py:42-54** [perf]
- _find_llama_cli is invoked per request alongside _find_model. It iterates a candidate list calling os.path.isfile + os.access(X_OK) on up to 6 paths each time. The binary location is effectively static for the process lifetime, so this is repeated, redundant filesystem I/O on the request path.
- **Fix:** The proposed fix is correct. Prefer the instance-attribute approach over functools.lru_cache: in __init__ add `self._llama_cli_cache: Optional[str] = None` and a sentinel to distinguish "not resolved" from "resolved to None", e.g. memoize as:

    def _find_llama_cli(self) -> Optional[str]:
        if not hasattr(self, "_llama_cli_resolved"):
            candidates = [
                os.environ.get("ARCFLARE_LLAMA_CLI", ""),
                "/usr/local/bin/llama-cli", "/usr/local/bin/llama",
                "/tmp/llama-cli", "/app/llama-cli", "/app/llama",
            ]
            self._llama_cli = next(
                (p for p in candidates if p and os.path.isfile(p) and os.access(p, os.X_OK)),
                None,
            )
            self._llama_cli_resolved = True
        return self._llama_cli

This caches a negative result (None) too, avoiding re-probing when the binary is absent. Note: since the singleton pipeline lives for the whole process, the cache persists correctly. Avoid lru_cache on the method (it keys on `self` and can complicate GC/testing). The same per-request redundancy applies to _find_model's os.listdir at lines 36-38, but that is out of scope for this specific finding.

### proc.communicate() buffers the full model output before any token is yielded (no real streaming)
- **/home/harty/projekt/orchestrator/src/arcflare/inference/pipeline.py:180-205 (RPC), 371-395 (local)** [perf]
- Both inference paths await proc.communicate() with a 600s timeout, which blocks until the subprocess has fully finished and produced ALL of stdout, then splits the complete buffer on newlines and yields lines with an artificial await asyncio.sleep(0.005)/0.01 between them. So despite the AsyncGenerator/SSE plumbing in openai.py, the client receives nothing until generation is 100% complete, and then gets the whole answer dribbled out with fake delays. Time-to-first-token equals total generation time, defeating the purpose of the streaming endpoints. The artificial sleeps also add latency proportional to output length (e.g. 0.01s * N lines).
- **Fix:** The actionable core of the proposed fix is correct: replace proc.communicate() with incremental reads — `while True: line = await proc.stdout.readline(); if not line: break; ...parse and yield...` — and concurrently drain stderr in a separate task to avoid pipe-buffer deadlock; remove the artificial asyncio.sleep(0.005/0.01) calls. Drop the 'persistent llama-server / stream=true' clause: it does not apply because this code invokes llama-cli (one-shot subprocess), not a server. Two correctness caveats the fix must address: (1) Preserve the existing parsing state machine when reading line-by-line — it waits for a line starting with '> ' to set capturing=True, breaks on lines starting with '[' or 'Exiting', and strips spinner/control chars ('|/-\\=', '\\b', '\\r'). These rules must be applied per-line incrementally, not lost. (2) llama-cli (like most CLI tools) full-buffers stdout when stdout is a pipe rather than a TTY, so readline() may still not flush per-token until the process exits — defeating the fix. To truly get incremental output you likely need to force unbuffered output (e.g. run under `stdbuf -oL`/`-o0`, or allocate a PTY via pty/os.openpty so llama-cli line-buffers as if interactive). Without that, switching to readline() alone will not reduce time-to-first-token. The long-term correct architecture is to run a persistent llama-server with an HTTP token-stream API, but that is a larger change than this finding's file scope.

### Shards reloaded on every gRPC request, with gguf-splitter subprocess spawned per request to count layers
- **/home/harty/projekt/orchestrator/src/arcflare/inference/pipeline.py:213-255, 270-271** [perf]
- _distributed_inference calls _load_shards_on_nodes on every request (line 271), which loops every node and issues a client.load_shard RPC each time -- re-sending/re-loading the model shard on the node for each inference even though the shard is unchanged. Worse, for every node it calls _get_model_layer_count(model_path) (line 222), which shells out to the gguf-splitter binary (subprocess.run with text parsing, 10s timeout) just to read the layer count -- a constant property of the model file -- and does so once per node per request (N subprocess spawns per request).
- **Fix:** The proposed fix is correct in direction; tighten the shard-loaded tracking to be topology-aware. (a) Cache the layer count per model_path: store on the instance (e.g. self._layer_count_cache: dict[str, int]) or @lru_cache the helper, or better read n_layers from the GGUF metadata header directly (e.g. *.block_count) instead of spawning gguf-splitter at all. (b) Track loaded shards keyed not just by (node_id, model) but by the full computed shard config — (node_id, model_path, first_layer, num_layers, has_lm_head) — because the per-node layer split is derived from len(nodes) and nodes.index(node); if the node set changes the previously-loaded shard range becomes invalid. Skip load_shard only when the node already holds exactly that shard config; evict/invalidate cache entries for a node when it drops out of node_connections or when the node set changes, so a topology change triggers a correct reload rather than reusing a wrongly-sized shard. (c) Minor: move `import subprocess` to module top instead of re-importing inside _get_model_layer_count on each call.

### O(n^2) nodes.index() calls inside the shard-loading loop
- **/home/harty/projekt/orchestrator/src/arcflare/inference/pipeline.py:225-227** [perf]
- Inside the per-node loop in _load_shards_on_nodes, the code calls nodes.index(node) three separate times per iteration (lines 225, 226, 227). list.index is O(n), so this is O(n^2) over the node list and also fragile (returns the first match if duplicate dicts exist). It is computing the loop index that enumerate already provides.
- **Fix:** Replace the loop header `for node in nodes:` (line 217) with `for idx, node in enumerate(nodes):`, then substitute `idx` for every `nodes.index(node)` call at lines 225-227:
    first = idx * layers_per
    num = layers_per if idx < n_nodes - 1 else total_layers - first
    has_head = idx == n_nodes - 1

### gRPC channels reconnected and connect attempted redundantly on the request path
- **/home/harty/projekt/orchestrator/src/arcflare/inference/pipeline.py:118, 270 / grpc_client connect 29-39** [perf]
- _connect_to_nodes is invoked from run_stream (line 118) and again from _distributed_inference (line 270) on every request. While it guards on node_id already in self.node_connections, it still iterates all nodes and constructs NodeGrpcClient + awaits client.connect() (grpc.aio.insecure_channel + a connectivity check) for any node not yet cached, on the hot path. There is no health/liveness reuse beyond presence-in-dict, and channels are never proactively warmed at startup, so the first request to each node pays full channel-establishment cost synchronously in the request.
- **Fix:** Two changes are needed; the original fix only covers the first:

1. Warm connections off the request path: in main.py startup (or in the discovery service callback when a node first appears), kick off a background task that calls a connect/refresh routine to populate self.node_connections, with periodic re-validation. Keep clients in the pool and reuse them.

2. Fix the bootstrap guard bug in run_stream (pipeline.py:114-118). Currently `active_connections = list(self.node_connections.keys())` and the branch only runs `if alive and active_connections:`, so a fresh node is never connected on-demand because the connect path requires the dict to already be non-empty. Change the guard to enter the branch when `alive` is non-empty (regardless of active_connections), let `_connect_to_nodes` populate the pool, and only fall back to local if no connection could be established. Otherwise the gRPC path is effectively dead code and the 'connect on request' cost the finding worries about never actually occurs.

3. Remove the duplicate `_connect_to_nodes` call: drop it from either run_stream (line 118) or _distributed_inference (line 270) so it runs at most once per request, and on the request path only validate cheap channel state (channel.get_state()) / re-warm asynchronously rather than re-running connect() (which makes a blocking GetHardwareInfo RPC, grpc_client.py:34) for the whole list.

If item 2 is not addressed, the perf concern is partly moot since the connect-on-request path is currently unreachable — but the code as written still has both the latent bug and the redundant double-call.

### get_nodes() / get_rpc_endpoints() deep-copy all node state via asdict on every call
- **/home/harty/projekt/orchestrator/src/arcflare/cluster/discovery.py:103-118** [perf]
- get_nodes() runs _prune_dead_nodes() and then builds a fresh asdict(n) (a recursive deep copy, including the hardware dict) for every node on every call. These are called on the inference hot path (run_stream lines 97-98) plus management/partition endpoints. dataclasses.asdict is notably slow because it deepcopies nested structures. For frequent polling or every inference request this is repeated allocation and copying of data that rarely changes.
- **Fix:** Replace asdict with a manual plain-dict builder; do not rely on the proposed cache. Add a helper, e.g. def _node_dict(n: NodeInfo) -> dict: return {"node_id": n.node_id, "node_name": n.node_name, "grpc_port": n.grpc_port, "version": n.version, "os": n.os, "last_seen": n.last_seen, "hardware": n.hardware, "status": n.status, "ip_address": n.ip_address, "rpc_port": n.rpc_port}, then get_nodes() returns [self._node_dict(n) for n in self.nodes.values()] and get_node() returns self._node_dict(node) if node else None. This removes the recursive copy.deepcopy that asdict performs on leaf values (the hardware dict is shared by reference, which is safe because the current consumers — pipeline.py:113 .get(), partition.py, management.py — read only). The finding's preferred cache option is the weaker choice here: a cache invalidated only on add/update/prune would still be stale because _handle_discovery mutates last_seen/status/rpc_port (discovery.py:95-98) on every heartbeat, which is frequent, so the cache would have to be invalidated on each heartbeat and would buy little. If callers later mutate the returned hardware dict, switch to dict(n.hardware) for a shallow copy, which is still far cheaper than asdict's deepcopy.

## NIT (7)

### Orchestrator registration failure is only logged, agent proceeds as if registered
- **node-agent/src/main.rs:118-129** [rust-agent]
- A non-success HTTP status or a network error from /api/nodes/register is only `tracing::warn!`-logged; the agent continues to serve and broadcast as though registered. Depending on intended semantics this may silently leave the node unregistered/orphaned with no retry. This is a deliberate best-effort design but worth flagging since failures are swallowed with no retry or backoff.
- **Fix:** The proposed fix is sound and complete (offers both fail-fast/retry-with-backoff if registration is required, or document + periodic re-registration if best-effort). One additional nuance worth adding: HTTP registration only runs when `--orchestrator-host` is supplied (line 106); when omitted, the node relies entirely on the UDP discovery broadcaster (lines 97-103) and never attempts HTTP registration at all. This reinforces that the design is intentionally best-effort. If a periodic re-registration loop is chosen, it should be spawned as a background task (e.g. tokio::spawn) so it does not block the gRPC serve loop, and ideally also cover the success-but-later-orchestrator-restart case, not just startup transient failures.

### GetHardwareInfo used as connectivity probe but channel stays open even when probe fails after channel creation
- **orchestrator/src/arcflare/inference/grpc_client.py:29-41** [py-pipeline]
- connect() creates self._channel, then awaits GetHardwareInfo; on exception it sets _channel/_stub to None but never calls await self._channel.close() before discarding it. The underlying grpc.aio channel object is abandoned without close, leaking the channel/its background tasks until GC, and grpc.aio may emit 'channel was not closed' warnings. Repeated failed connects (dead node still advertised) leak one channel each.
- **Fix:** In the except block, close the channel before nulling it, and guard the close itself so a close failure cannot mask the original error or escape connect(). Replace lines 39-40:

        except Exception as e:
            logger.warning(f"Failed to connect to node {self.node_id} at {self.address}: {e}")
            if self._channel is not None:
                try:
                    await self._channel.close()
                except Exception:
                    pass
            self._channel = None
            self._stub = None
            return False

The finding's proposed fix (await self._channel.close() guarded by a None check) is correct in substance; the only addition is wrapping close() in its own try/except so a secondary failure during cleanup does not propagate out of connect(), which is expected to return False rather than raise.

### Prompt passed to llama-cli is safe from shell injection but flag-injection is possible
- **orchestrator/src/arcflare/inference/pipeline.py:158-166, 355-362** [py-pipeline]
- create_subprocess_exec is used (not shell=True), so there is no shell injection via the prompt — good. However the prompt is placed in the argv after -p, and there is no '--' end-of-options separator. While -p takes the next token as its value (so a leading '-' in the prompt is consumed as the value), other unguarded user-influenced values are not delimited; more importantly model_path/llama_cli are derived from environment variables (ARCFLARE_MODELS_DIR, ARCFLARE_LLAMA_CLI) without validation. A path beginning with '-' from env could be interpreted as a flag. Low severity given env is operator-controlled, but worth hardening.
- **Fix:** The genuine, defensible hardening (defense-in-depth, nit severity) is to validate the env-derived llama_cli before exec'ing it, since it is used unvalidated as argv[0]:

1. In _find_llama_cli (lines 42-54): the ARCFLARE_LLAMA_CLI value is already gated by os.path.isfile + os.access(os.X_OK), which is reasonable. Optionally normalize it with os.path.abspath() for consistency with _find_model and to guarantee it never begins with "-".

2. Drop the "-p"/model_path flag-injection rationale: model_path is already absolute (os.path.abspath at line 34, then os.path.join at line 38), so it cannot start with "-". And llama_cli is argv[0], which is not flag-parsed. There is no actual reachable flag-injection here.

3. If a "--" terminator is desired purely as belt-and-suspenders, note that llama.cpp's llama-cli does NOT support a generic "--" positional terminator the same way GNU tools do (it has no positional args), and "-p" already binds the next single token regardless of a leading "-", so adding "--" gives no real benefit for the prompt.

Recommended concrete change: in _find_model, optionally also guard against a resolved model file accidentally beginning with "-" by prefixing relative results — but since abspath already runs, this is unnecessary. The only material change is to keep using create_subprocess_exec (already done) and optionally abspath() the llama_cli path. This finding is genuinely a nit and arguably could be marked won't-fix given the current code already prevents the described exploit.

### format_messages silently drops messages with unknown roles and uses a non-standard prompt template
- **/home/harty/projekt/orchestrator/src/arcflare/api/openai.py:184-193** [py-api]
- format_messages only handles 'system'/'user'/'assistant'; any other role (e.g. 'tool', 'function', or a typo) is silently discarded, losing context. It also builds a generic 'User:/Assistant:' transcript and does not append a trailing 'Assistant:' turn to cue the model to continue, nor does it apply the model's actual chat template — for instruction-tuned GGUF models this degrades output quality. There is no handling of empty messages list (would send an empty prompt).
- **Fix:** The proposed fix is valid. One refinement: the non-empty validation belongs in the endpoint handlers (chat_completions and generate_chat_stream both call format_messages), not inside the pure formatting helper — or enforce it declaratively on the model so both paths and FastAPI's 422 machinery cover it. Concretely: add `messages: list[ChatMessage] = Field(min_length=1)` to ChatCompletionRequest so Pydantic rejects an empty list with a 422 automatically (covering both the streaming and non-streaming paths), rather than raising HTTPException from within format_messages. Then in format_messages add the unknown-role fallback `else: parts.append(f"{msg.role}: {msg.content}")` and append a trailing cue after the loop, e.g. `parts.append("Assistant:")` (note: bare 'Assistant:' with no space-content, joined by '\n', so the model continues that turn). Best long-term fix remains applying the model's real chat template in the inference layer (inference/pipeline.run_inference) instead of this hand-rolled transcript, since the current approach also ignores per-model prompt formats entirely.

### asyncio.create_task return value discarded — discovery start task can be GC'd and errors swallowed
- **/home/harty/projekt/orchestrator/src/arcflare/main.py:23** [py-api]
- asyncio.create_task(discovery_service.start()) is not assigned to anything. Per CPython docs the event loop only keeps a weak reference to tasks, so a fire-and-forget task may be garbage-collected before completion, and any exception raised inside start() (other than the caught OSError) is lost with no traceback. Since start() is short-lived and mostly awaits create_datagram_endpoint this is low-impact, but it is a known footgun.
- **Fix:** Prefer the await-directly option in lifespan, since start() already handles its only expected failure (OSError) internally and is quick:

    discovery_service = DiscoveryService()
    await discovery_service.start()
    logger.info("Discovery service started on UDP port 5678")
    yield

This eliminates the discarded-task/GC issue and surfaces any unexpected exceptions during startup instead of losing them. (If a non-blocking background task is ever genuinely needed, the alternative is to retain a reference, e.g. `app.state.discovery_task = asyncio.create_task(...)`, but that alone would not surface swallowed exceptions without also attaching a done-callback.)

### os.listdir model scan on every request in _find_model
- **/home/harty/projekt/orchestrator/src/arcflare/inference/pipeline.py:29-40** [perf]
- _find_model is called on every request from _rpc_distributed_inference (line 147), _distributed_inference (line 265) and _local_inference (line 339). Each call does os.path.abspath + os.listdir(models_dir) and string-scans the directory listing to find the first *.gguf file. This is a syscall-heavy directory walk on the hot path that returns the same answer every time (the model set does not change between requests).
- **Fix:** Cache the resolved path as an instance attribute populated lazily, with a stat-based existence check to invalidate if the file disappears:

```python
def __init__(self):
    self.active_pipelines: dict = {}
    self.node_connections: dict[str, NodeGrpcClient] = {}
    self._model_path: Optional[str] = None

def _find_model(self, model_name: str) -> Optional[str]:
    if self._model_path and os.path.isfile(self._model_path):
        return self._model_path
    models_dir = os.environ.get(
        "ARCFLARE_MODELS_DIR",
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "models"),
    )
    models_dir = os.path.abspath(models_dir)
    for fname in os.listdir(models_dir):
        if fname.endswith(".gguf"):
            self._model_path = os.path.join(models_dir, fname)
            return self._model_path
    self._model_path = None
    return None
```

Do NOT use the proposed `functools.lru_cache` keyed on `(models_dir, model_name)`: (1) it is an instance method so the cache would key on `self` and retain instances, and (2) `model_name` is irrelevant — the current implementation ignores it entirely and always returns the first `*.gguf` found, so keying on it would be misleading and would not match behavior. The instance-attribute approach is correct and avoids these issues. Note the residual `os.path.isfile` stat is itself a syscall, so the net win is replacing a listdir+scan with a single stat — minor, consistent with the nit severity.

### Generated arcflare_pb2.py is stale vs proto/arcflare.proto — missing RpcEndpoint message and GetRpcEndpoint RPC
- **/home/harty/projekt/orchestrator/src/arcflare/arcflare_pb2.py:1** [deploy-proto]
- The committed generated Python stubs are out of sync with proto/arcflare.proto. The proto defines `message RpcEndpoint` and `rpc GetRpcEndpoint(Empty) returns (RpcEndpoint)` (lines 32, 280-284), and the Rust server implements `get_rpc_endpoint`. But the generated pb2 is missing both `RpcEndpoint` and `GetRpcEndpoint` (verified: those tokens are absent from arcflare_pb2.py, while `rpc_port`, `session_token`, `ForwardStream`, `GetInferenceStats` are present). This is latent rather than fatal because the current Python client (inference/grpc_client.py) never calls GetRpcEndpoint — the orchestrator gets rpc_port via UDP discovery / HTTP register instead. Note: protobuf runtime is fine (pb2 requires protobuf>=6.33.5, and grpcio-tools in requirements pins protobuf>=6.33.5,<7), so the version check passes. Regenerate to avoid surprises if anyone wires up GetRpcEndpoint on the client.
- **Fix:** The proposed fix is correct and complete. Regenerate both stubs and restore the package-relative import:

  python -m grpc_tools.protoc -I proto \
    --python_out=orchestrator/src/arcflare \
    --grpc_python_out=orchestrator/src/arcflare \
    proto/arcflare.proto

Then in the regenerated orchestrator/src/arcflare/arcflare_pb2_grpc.py change the emitted top-level `import arcflare_pb2 as arcflare__pb2` back to `from . import arcflare_pb2 as arcflare__pb2` (the committed file already uses the relative form; fresh protoc output drops it). Note regeneration will also (correctly) add the currently-missing `RegisterRequest.rpc_port` field, not just RpcEndpoint/GetRpcEndpoint — the existing stubs are stale on that field too.

