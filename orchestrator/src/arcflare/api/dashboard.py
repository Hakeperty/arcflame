"""Phase 3 — web dashboard.

Serves a single self-contained HTML page (vanilla JS, no build step) that polls
the management API for live cluster state and provides an inference tester.
Mounted at GET /dashboard.
"""
from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter(tags=["Dashboard"])

_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ArcFlare — Cluster Dashboard</title>
<style>
  :root { --bg:#0d1117; --panel:#161b22; --border:#30363d; --fg:#c9d1d9; --accent:#f78166; --green:#3fb950; --muted:#8b949e; }
  * { box-sizing: border-box; }
  body { margin:0; font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif; background:var(--bg); color:var(--fg); }
  header { padding:16px 24px; border-bottom:1px solid var(--border); display:flex; align-items:center; gap:14px; }
  header h1 { font-size:20px; margin:0; }
  header h1 .flare { color:var(--accent); }
  .pill { font-size:12px; padding:3px 10px; border-radius:999px; border:1px solid var(--border); color:var(--muted); }
  .pill.live { color:var(--green); border-color:var(--green); }
  main { padding:24px; max-width:1100px; margin:0 auto; }
  .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:14px; margin-bottom:24px; }
  .card { background:var(--panel); border:1px solid var(--border); border-radius:10px; padding:16px; }
  .card .label { font-size:12px; color:var(--muted); text-transform:uppercase; letter-spacing:.05em; }
  .card .value { font-size:28px; font-weight:600; margin-top:6px; }
  .card .value.mode { font-size:18px; color:var(--accent); }
  h2 { font-size:14px; color:var(--muted); text-transform:uppercase; letter-spacing:.05em; margin:24px 0 10px; }
  table { width:100%; border-collapse:collapse; background:var(--panel); border:1px solid var(--border); border-radius:10px; overflow:hidden; }
  th,td { text-align:left; padding:10px 14px; border-bottom:1px solid var(--border); font-size:13px; }
  th { color:var(--muted); font-weight:600; background:#11161d; }
  tr:last-child td { border-bottom:none; }
  .dot { display:inline-block; width:8px; height:8px; border-radius:50%; background:var(--green); margin-right:6px; }
  .dot.stale { background:var(--muted); }
  .empty { color:var(--muted); padding:18px; text-align:center; }
  .chat { background:var(--panel); border:1px solid var(--border); border-radius:10px; padding:16px; }
  textarea { width:100%; min-height:64px; background:var(--bg); color:var(--fg); border:1px solid var(--border); border-radius:8px; padding:10px; font-family:inherit; font-size:14px; resize:vertical; }
  .row { display:flex; gap:10px; align-items:center; margin-top:10px; }
  button { background:var(--accent); color:#0d1117; border:none; border-radius:8px; padding:9px 18px; font-weight:600; cursor:pointer; font-size:14px; }
  button:disabled { opacity:.5; cursor:default; }
  #answer { margin-top:14px; padding:14px; background:var(--bg); border:1px solid var(--border); border-radius:8px; white-space:pre-wrap; min-height:24px; font-size:14px; }
  .muted { color:var(--muted); font-size:12px; }
  code { color:var(--accent); }
</style>
</head>
<body>
<header>
  <h1>Arc<span class="flare">Flare</span></h1>
  <span class="pill" id="conn">connecting…</span>
  <span class="pill" id="mode">—</span>
</header>
<main>
  <div class="grid">
    <div class="card"><div class="label">Status</div><div class="value" id="m-status">—</div></div>
    <div class="card"><div class="label">Nodes</div><div class="value" id="m-nodes">—</div></div>
    <div class="card"><div class="label">Total RAM</div><div class="value" id="m-ram">—</div></div>
    <div class="card"><div class="label">GPUs</div><div class="value" id="m-gpus">—</div></div>
    <div class="card"><div class="label">Pipeline</div><div class="value mode" id="m-mode">—</div></div>
  </div>

  <h2>Nodes</h2>
  <table>
    <thead><tr><th>Name</th><th>Node ID</th><th>Address</th><th>gRPC</th><th>RPC</th><th>OS</th></tr></thead>
    <tbody id="nodes"><tr><td colspan="6" class="empty">No nodes registered yet.</td></tr></tbody>
  </table>

  <h2>RPC Endpoints</h2>
  <div id="rpc" class="muted">—</div>

  <h2>Inference Tester</h2>
  <div class="chat">
    <textarea id="prompt" placeholder="Ask the cluster something…">What is the capital of France? One word.</textarea>
    <div class="row">
      <button id="send">Send</button>
      <span class="muted">POST <code>/v1/chat/completions</code> → model <code>arcflare/default</code></span>
    </div>
    <div id="answer" class="muted">Response will appear here.</div>
  </div>
</main>
<script>
const $ = (id) => document.getElementById(id);

async function refresh() {
  try {
    const r = await fetch('/api/cluster/status');
    const d = await r.json();
    $('conn').textContent = 'live'; $('conn').className = 'pill live';
    $('m-status').textContent = d.status ?? '—';
    $('m-nodes').textContent = d.nodes ?? 0;
    $('m-ram').textContent = (d.total_ram_gb ? d.total_ram_gb.toFixed(1) : '0') + ' GB';
    $('m-gpus').textContent = d.total_gpus ?? 0;
    $('m-mode').textContent = d.pipeline_mode ?? '—';
    $('mode').textContent = d.pipeline_mode ?? '—';
    const eps = d.rpc_endpoints || [];
    $('rpc').textContent = eps.length ? eps.join('  ·  ') : 'none — running in ' + (d.pipeline_mode || 'local') + ' mode';
  } catch (e) {
    $('conn').textContent = 'offline'; $('conn').className = 'pill';
  }
  try {
    const r = await fetch('/api/nodes');
    const { nodes = [] } = await r.json();
    const tb = $('nodes');
    if (!nodes.length) { tb.innerHTML = '<tr><td colspan="6" class="empty">No nodes registered yet.</td></tr>'; return; }
    tb.innerHTML = nodes.map(n => {
      const alive = (n.status === 'alive' || n.status === 'discovered');
      const ip = n.ip_address || '—';
      return `<tr>
        <td><span class="dot ${alive ? '' : 'stale'}"></span>${n.node_name ?? n.name ?? '—'}</td>
        <td class="muted">${(n.node_id||'').slice(0,24)}</td>
        <td>${ip}</td>
        <td>${n.grpc_port ?? '—'}</td>
        <td>${n.rpc_port ? n.rpc_port : '<span class="muted">off</span>'}</td>
        <td class="muted">${n.os ?? '—'}</td>
      </tr>`;
    }).join('');
  } catch (e) {}
}

$('send').addEventListener('click', async () => {
  const btn = $('send'), out = $('answer');
  const content = $('prompt').value.trim();
  if (!content) return;
  btn.disabled = true; out.className = ''; out.textContent = 'Thinking…';
  const t0 = performance.now();
  try {
    const r = await fetch('/v1/chat/completions', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ model:'arcflare/default', messages:[{role:'user', content}], max_tokens:64 })
    });
    const d = await r.json();
    const txt = d?.choices?.[0]?.message?.content ?? JSON.stringify(d);
    const secs = ((performance.now()-t0)/1000).toFixed(1);
    out.textContent = txt + `\n\n— ${secs}s`;
  } catch (e) {
    out.textContent = 'Error: ' + e;
  } finally { btn.disabled = false; }
});

refresh();
setInterval(refresh, 3000);
</script>
</body>
</html>"""


@router.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
async def dashboard() -> HTMLResponse:
    return HTMLResponse(_PAGE)
