"""
tune.py - live parameter tuning in the browser: BEFORE (raw min/max) next to AFTER (ISP).

  python tune.py data.bin                    # opens http://127.0.0.1:8765
  python tune.py data.bin --config tuned.json --port 8800 --out out

Every parameter has a slider AND a number box. Changes apply to the next frame.
Buttons: play/pause, step, restart pipeline, save config (JSON), export the two videos
(<name>_before.mp4, <name>_after.mp4 and <name>_compare.mp4) with the current settings.
Only the Python standard library + numpy/opencv are used (no extra installs).
"""
import argparse
import dataclasses
import inspect
import json
import os
import re
import threading
import time
import webbrowser
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

import numpy as np
import cv2

from isp import (Config, ThermalISP, naive_scale, open_frames, WIDTH, HEIGHT, AGC_MODES,
                 PARAM_RANGES, set_param, save_config, load_config)
from isp import config as config_module
from run import export_videos

WARMUP_FRAMES = 12   # frames run silently after a jump, so temporal/AGC state is settled

# fields that only make sense after a full pipeline restart
RESTART_FIELDS = {"badpix", "stripes", "temporal", "motion_compensation"}

GROUPS = [
    ("Stages", ["badpix", "stripes", "temporal", "motion_compensation", "dde"]),
    ("1 · Bad pixels", ["spike_sigmas"]),
    ("2 · Stripes", ["column_learn_rate"]),
    ("3 · Offset + temporal denoise", ["offset_release_rate", "still_weight"]),
    ("4 · Base / detail split", ["guided_radius", "edge_sigmas"]),
    ("5 · AGC", ["agc_mode", "min_range_sigmas", "out_low", "out_high", "scene_change_share"]),
    ("5a · Absolute (v2)", ["black_percentile", "white_percentile", "peak_percentile", "black_rate",
                            "white_attack", "white_release", "white_floor_sigmas", "knee", "eq_share", "gamma"]),
    ("5b · Large warm areas (optional)", ["warm_suppress", "warm_suppress_strength", "warm_suppress_size"]),
    ("5c · Plateau (v1) / percentile / mean", ["plateau_factor", "linear_share", "clip_low", "clip_high",
                 "mean_std_factor", "lut_smoothing", "lut_smoothing_fast"]),
    ("6 · Detail", ["detail_gain"]),
]


def field_help():
    """The one-line comments next to each field in config.py, used as tooltips."""
    help_text = {}
    for line in inspect.getsource(config_module.Config).splitlines():
        match = re.match(r"\s*(\w+)\s*:\s*\w+\s*=\s*[^#]+#\s*(.+)", line)
        if match:
            help_text[match.group(1)] = match.group(2).strip()
    return help_text


class Session:
    """The pipeline the browser is looking at. All access goes through `lock`."""

    def __init__(self, path, cfg, out_dir, fps, width, height):
        self.path, self.out_dir, self.fps = path, out_dir, fps
        self.width, self.height = width, height
        self.frames = open_frames(path, width, height)
        self.cfg = cfg
        self.lock = threading.Lock()
        self.isp = None
        self.last_index = -10
        self.last_ms = 0.0
        self.export = {"running": False, "done": 0, "total": 0, "files": [], "error": ""}

    def restart(self):
        self.isp = ThermalISP(self.cfg, self.width, self.height)
        self.last_index = -10

    def render(self, index, reprocess=False):
        """Side-by-side PNG of frame `index`. Sequential frames reuse the running state;
        a jump (or a re-render after a parameter change) restarts and warms up first."""
        index = int(np.clip(index, 0, len(self.frames) - 1))
        with self.lock:
            sequential = self.isp is not None and index == self.last_index + 1 and not reprocess
            if not sequential:
                self.restart()
                for warm in range(max(0, index - WARMUP_FRAMES), index):
                    self.isp.process(np.array(self.frames[warm]))
            raw = np.array(self.frames[index])
            start = time.perf_counter()
            after = self.isp.process(raw)
            self.last_ms = (time.perf_counter() - start) * 1000
            self.last_index = index
        before = naive_scale(raw)
        ok, png = cv2.imencode(".png", np.hstack([before, after]), [cv2.IMWRITE_PNG_COMPRESSION, 1])
        return png.tobytes()

    def set(self, name, value):
        with self.lock:
            value = set_param(self.cfg, name, value)
            if name == "agc_mode" and self.isp is not None:
                self.isp.agc.lut = None            # new mode at once, no slow fade
            if name in RESTART_FIELDS:
                self.isp = None
        return value

    def start_export(self):
        if self.export["running"]:
            return
        cfg = dataclasses.replace(self.cfg)
        self.export = {"running": True, "done": 0, "total": len(self.frames), "files": [], "error": ""}

        def progress(done, total):
            self.export["done"], self.export["total"] = done, total

        def work():
            try:
                files = export_videos(self.path, self.out_dir, self.fps, cfg, True, progress=progress,
                                      width=self.width, height=self.height)
                save_config(cfg, os.path.join(self.out_dir, "used_config.json"))
                self.export["files"] = [os.path.abspath(f) for f in files]
            except Exception as error:          # shown in the page
                self.export["error"] = str(error)
            self.export["running"] = False

        threading.Thread(target=work, daemon=True).start()


def make_handler(session, page):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, body, kind="application/json", code=200, headers=None):
            if isinstance(body, (dict, list)):
                body = json.dumps(body).encode()
            elif isinstance(body, str):
                body = body.encode()
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for key, value in (headers or {}).items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body)

        def state(self):
            cfg = dataclasses.asdict(session.cfg)
            return {"config": cfg, "ranges": PARAM_RANGES, "modes": AGC_MODES, "groups": GROUPS,
                    "help": field_help(), "frames": len(session.frames),
                    "file": os.path.basename(session.path), "out": os.path.abspath(session.out_dir)}

        def do_GET(self):
            url = urlparse(self.path)
            query = parse_qs(url.query)
            if url.path == "/":
                self.send(page, "text/html; charset=utf-8")
            elif url.path == "/state":
                self.send(self.state())
            elif url.path == "/frame":
                png = session.render(int(query.get("i", ["0"])[0]), query.get("re", ["0"])[0] == "1")
                self.send(png, "image/png", headers={"X-Ms": f"{session.last_ms:.1f}"})
            elif url.path == "/export":
                self.send(session.export)
            elif url.path == "/favicon.ico":
                self.send(b"", "image/x-icon", code=204)
            elif url.path == "/config.json":
                self.send(json.dumps(dataclasses.asdict(session.cfg), indent=2), "application/json",
                          headers={"Content-Disposition": "attachment; filename=tuned.json"})
            else:
                self.send({"error": "not found"}, code=404)

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(length) or b"{}")
            url = urlparse(self.path).path
            try:
                if url == "/set":
                    value = session.set(data["name"], data["value"])
                    self.send({"name": data["name"], "value": value})
                elif url == "/reset":
                    with session.lock:
                        session.cfg = Config()
                        session.isp = None
                    self.send(self.state())
                elif url == "/load":
                    cfg = Config()
                    for name, value in data.items():
                        set_param(cfg, name, value)
                    with session.lock:
                        session.cfg = cfg
                        session.isp = None
                    self.send(self.state())
                elif url == "/save":
                    os.makedirs(session.out_dir, exist_ok=True)
                    target = os.path.join(session.out_dir, "tuned.json")
                    save_config(session.cfg, target)
                    self.send({"saved": os.path.abspath(target)})
                elif url == "/export":
                    session.start_export()
                    self.send(session.export)
                else:
                    self.send({"error": "not found"}, code=404)
            except (KeyError, ValueError) as error:
                self.send({"error": str(error)}, code=400)

    return Handler


PAGE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>Thermal ISP tuner</title>
<style>
:root{--bg:#111316;--panel:#1a1d21;--line:#2a2e34;--text:#e6e6e6;--dim:#8b929b;--acc:#f0b429}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font:13px/1.4 system-ui,sans-serif;display:flex;height:100vh}
main{flex:1;display:flex;flex-direction:column;min-width:0;padding:12px;gap:8px}
aside{width:380px;background:var(--panel);border-left:1px solid var(--line);overflow-y:auto;padding:10px 14px}
.view{flex:1;display:flex;flex-direction:column;justify-content:center;min-height:0}
.labels{display:flex}.labels div{flex:1;color:var(--dim);font-size:12px;letter-spacing:.06em;text-transform:uppercase;padding:2px 4px}
#img{width:100%;max-height:calc(100vh - 150px);object-fit:contain;image-rendering:pixelated;background:#000}
.bar{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
button{background:#262a30;color:var(--text);border:1px solid var(--line);border-radius:4px;padding:5px 10px;cursor:pointer;font:inherit}
button:hover{border-color:var(--dim)} button.main{background:var(--acc);color:#111;border-color:var(--acc);font-weight:600}
#seek{flex:1;min-width:200px} .stat{color:var(--dim);font-variant-numeric:tabular-nums}
h3{margin:14px 0 6px;font-size:12px;color:var(--acc);text-transform:uppercase;letter-spacing:.06em}
.row{display:grid;grid-template-columns:130px 1fr 72px;gap:6px;align-items:center;margin:4px 0}
.row label{color:var(--text);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;cursor:help}
.row input[type=number],select{width:100%;background:#0f1114;color:var(--text);border:1px solid var(--line);border-radius:3px;padding:3px 4px;font:inherit}
.check{display:flex;gap:6px;align-items:center;margin:3px 0;cursor:help}
input[type=range]{width:100%;accent-color:var(--acc)}
#msg{color:var(--dim);min-height:1.4em;word-break:break-all}
</style></head><body>
<main>
  <div class="bar"><b id="title">Thermal ISP</b><span class="stat" id="stat"></span></div>
  <div class="view">
    <div class="labels"><div>Before · raw min/max</div><div>After · ISP white-hot</div></div>
    <img id="img" alt="before / after">
  </div>
  <div class="bar">
    <button id="play" class="main">❚❚ Pause</button>
    <button id="prev">◀ Step</button><button id="next">Step ▶</button>
    <input id="seek" type="range" min="0" value="0">
    <span class="stat" id="pos"></span>
  </div>
  <div class="bar">
    <button id="export" class="main">Export before / after videos</button>
    <button id="save">Save config</button>
    <a id="dl" href="/config.json"><button>Download config</button></a>
    <label><button id="loadbtn">Load config…</button><input id="load" type="file" accept=".json" hidden></label>
    <button id="reset">Defaults</button>
    <button id="restart">Restart pipeline</button>
  </div>
  <div id="msg"></div>
</main>
<aside id="panel"></aside>
<script>
let S, idx = 0, playing = true, busy = false, dirty = false, timer = null;
const $ = id => document.getElementById(id);
const msg = t => $("msg").textContent = t;

async function api(path, body) {
  const r = await fetch(path, body === undefined ? {} : {method: "POST", body: JSON.stringify(body)});
  const j = await r.json(); if (!r.ok) throw new Error(j.error || r.status); return j;
}

function decimals(step) { const s = String(step); return s.includes(".") ? s.split(".")[1].length : 0; }

function buildPanel() {
  const panel = $("panel"); panel.innerHTML = "";
  for (const [group, names] of S.groups) {
    const h = document.createElement("h3"); h.textContent = group; panel.appendChild(h);
    for (const name of names) {
      const value = S.config[name], tip = S.help[name] || "";
      if (typeof value === "boolean") {
        const l = document.createElement("label"); l.className = "check"; l.title = tip;
        l.innerHTML = `<input type="checkbox" ${value ? "checked" : ""}> ${name}`;
        l.querySelector("input").onchange = e => send(name, e.target.checked);
        panel.appendChild(l); continue;
      }
      const row = document.createElement("div"); row.className = "row";
      const lab = `<label title="${name}: ${tip}">${name}</label>`;
      if (name === "agc_mode") {
        row.innerHTML = lab + `<select>${S.modes.map(m => `<option ${m === value ? "selected" : ""}>${m}</option>`).join("")}</select><span></span>`;
        row.querySelector("select").onchange = e => send(name, e.target.value);
      } else {
        const [lo, hi, step] = S.ranges[name];
        row.innerHTML = lab + `<input type="range" min="${lo}" max="${hi}" step="${step}" value="${value}">` +
                        `<input type="number" min="${lo}" max="${hi}" step="${step}" value="${value}">`;
        const [slider, box] = row.querySelectorAll("input");
        slider.oninput = () => { box.value = (+slider.value).toFixed(decimals(step)); };
        slider.onchange = () => send(name, slider.value, [slider, box]);
        box.onchange = () => send(name, box.value, [slider, box]);
      }
      panel.appendChild(row);
    }
  }
}

async function send(name, value, inputs) {
  try {
    const r = await api("/set", {name, value});
    S.config[name] = r.value;
    if (inputs) inputs.forEach(i => i.value = r.value);
    msg(`${name} = ${r.value}`);
    if (!playing) show(idx, true); else dirty = true;
  } catch (e) { msg("Error: " + e.message); }
}

let pending = null;   // a request made while a frame was loading: run it next, never drop it
async function show(i, reprocess) {
  if (busy) { pending = [i, reprocess || (pending && pending[1])]; return; }
  busy = true;
  try {
    const r = await fetch(`/frame?i=${i}&re=${reprocess ? 1 : 0}`);
    const blob = await r.blob(), old = $("img").src;
    $("img").src = URL.createObjectURL(blob); if (old) URL.revokeObjectURL(old);
    idx = i; $("seek").value = i; $("pos").textContent = `frame ${i} / ${S.frames - 1}`;
    const ms = +r.headers.get("X-Ms");
    $("stat").textContent = `${ms.toFixed(1)} ms/frame · ${ms < 40 ? "within" : "over"} 40 ms budget`;
  } finally {
    busy = false;
    if (pending) { const [pi, pr] = pending; pending = null; show(pi, pr); }
  }
}

async function loop() {
  if (playing && !busy) { await show((idx + 1) % S.frames, false); }
  timer = setTimeout(loop, 40);
}

function setPlaying(p) { playing = p; $("play").textContent = p ? "❚❚ Pause" : "▶ Play"; }

async function pollExport() {
  const e = await api("/export");
  if (e.error) { msg("Export failed: " + e.error); $("export").disabled = false; return; }
  if (e.running) { msg(`Exporting… ${e.done} / ${e.total} frames`); setTimeout(pollExport, 500); return; }
  $("export").disabled = false;
  msg("Written: " + e.files.join("  ·  "));
}

async function refreshState(s) { S = s; buildPanel(); if (!playing) show(idx, true); }

$("play").onclick = () => setPlaying(!playing);
$("next").onclick = () => { setPlaying(false); show(Math.min(idx + 1, S.frames - 1), false); };
$("prev").onclick = () => { setPlaying(false); show(Math.max(idx - 1, 0), true); };
$("seek").oninput = e => { setPlaying(false); };
$("seek").onchange = e => show(+e.target.value, true);
$("restart").onclick = () => show(idx, true);
$("reset").onclick = async () => { await refreshState(await api("/reset", {})); msg("Defaults restored"); };
$("save").onclick = async () => { const r = await api("/save", {}); msg("Saved " + r.saved); };
$("loadbtn").onclick = e => { e.preventDefault(); $("load").click(); };
$("load").onchange = async e => {
  try { const data = JSON.parse(await e.target.files[0].text());
        await refreshState(await api("/load", data)); msg("Config loaded"); }
  catch (err) { msg("Load failed: " + err.message); }
  e.target.value = "";
};
$("export").onclick = async () => { $("export").disabled = true; await api("/export", {}); pollExport(); };
document.addEventListener("keydown", e => {
  if (e.target.tagName === "INPUT" && e.target.type === "number") return;
  if (e.key === " ") { e.preventDefault(); setPlaying(!playing); }
  if (e.key === "d" || e.key === "ArrowRight") $("next").click();
  if (e.key === "a" || e.key === "ArrowLeft") $("prev").click();
});

(async () => {
  S = await api("/state");
  $("title").textContent = `Thermal ISP · ${S.file}`;
  $("seek").max = S.frames - 1;
  buildPanel(); await show(0, true); loop();
})();
</script></body></html>
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bin", help="raw 14-bit .bin sequence")
    parser.add_argument("--config", help="start from a saved JSON config")
    parser.add_argument("--out", default="out", help="where exported videos and configs go")
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--width", type=int, default=WIDTH)
    parser.add_argument("--height", type=int, default=HEIGHT)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else Config()
    session = Session(args.bin, cfg, args.out, args.fps, args.width, args.height)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(session, PAGE))
    url = f"http://127.0.0.1:{args.port}"
    print(f"{len(session.frames)} frames from {args.bin}\nTuner running at {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
