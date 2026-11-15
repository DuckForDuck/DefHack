"""Interactive tuner in the browser: change parameters, see the frame at once.

Run:  python tuner.py                  (page opens at http://127.0.0.1:8765)
      python tuner.py --params NAME    (start from a saved preset)

- Steps 0–3 are computed once for the whole recording (≈ 12 s) and cached.
- Steps 5–7 are re-run for the shown frame whenever a value changes.
- Settings are saved as named presets in presets/<name>.json;
  main.py --params <name> uses one.

Keys: A / D frame back / forward (Shift: 10 frames), Space play/pause,
Up / Down select a parameter, Left / Right change it (Shift: 10 steps),
R reset the selected parameter.

Only the Python standard library is used for the server; it listens on
127.0.0.1 only (this computer).
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2

from main import (KEY_FRAMES, PRESETS_DIR, TUNABLE, add_source_args,
                  build_back, build_front)
from thermal import Pipeline, RawFileSource
from thermal.stages import Decode, NaiveAGC
from thermal.tuning import (TuningSession, list_presets, load_params,
                            preset_path, resolve_preset, save_params)


def png_b64(img) -> str:
    ok, buf = cv2.imencode(".png", img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    return base64.b64encode(buf.tobytes()).decode("ascii")


class Handler(BaseHTTPRequestHandler):
    session: TuningSession = None   # set in main()
    preset: str | None = None       # name of the loaded / last saved preset
    preset_values: dict = {}        # its values, to show "modified"

    def log_message(self, fmt, *args):  # keep the console quiet
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False).encode(), "application/json")

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def _state(self) -> dict:
        s, cls = self.session, type(self)
        values = s.values()
        modified = cls.preset is not None and any(
            abs(float(values[k]) - float(cls.preset_values.get(k, values[k]))) > 1e-9
            for k in values)
        return {"params": s.spec(), "values": values, "defaults": s.defaults,
                "n": s.n, "indices": s.indices, "status": s.status,
                "front_dirty": s.front_dirty, "key_frames": KEY_FRAMES,
                "preset": cls.preset, "preset_modified": modified,
                "presets": list_presets(PRESETS_DIR),
                "presets_dir": PRESETS_DIR.name}

    def do_GET(self):
        url = urlparse(self.path)
        try:
            if url.path == "/":
                self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            elif url.path == "/api/state":
                self._json(self._state())
            elif url.path == "/api/render":
                k = int(parse_qs(url.query).get("k", ["0"])[0])
                r = self.session.render(k)
                self._json({"left": png_b64(r["left"]),
                            "right": png_b64(r["right"]) if r["right"] is not None else None,
                            "info": r["info"]})
            else:
                self._send(404, b"not found", "text/plain")
        except Exception as e:  # report errors to the page
            self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    def do_POST(self):
        url = urlparse(self.path)
        s, cls = self.session, type(self)
        try:
            body = self._body()
            if url.path == "/api/params":
                s.set_values(body.get("values", {}))
            elif url.path == "/api/reset":
                s.set_values(s.defaults)
            elif url.path == "/api/rebuild":
                s.build_front()
            elif url.path == "/api/save":
                path = preset_path(PRESETS_DIR, body.get("name", ""))
                if path.exists() and not body.get("overwrite"):
                    self._json({"exists": path.stem})
                    return
                save_params(path, s.values(), note=body.get("note", ""))
                cls.preset, cls.preset_values = path.stem, s.values()
            elif url.path == "/api/load":
                path = resolve_preset(PRESETS_DIR, body.get("name", ""))
                values = load_params(path)
                s.set_values(s.defaults)   # keys missing in the preset = default
                s.set_values(values)
                cls.preset, cls.preset_values = path.stem, s.values()
            else:
                self._send(404, b"not found", "text/plain")
                return
            self._json(self._state())
        except Exception as e:
            self._json({"error": f"{type(e).__name__}: {e}"}, 500)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_source_args(ap)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--warmup", type=int, default=40,
                    help="previous frames used to prepare the tone-map state")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    source = RawFileSource(args.input, width=args.width, height=args.height,
                           fps=args.fps, start=args.start, stop=args.stop)
    front, back = build_front(args), build_back()
    reference = Pipeline([Decode(bit_depth=14, stats=False), NaiveAGC()], profile=False)
    # defaults = values in the code (taken when the session is created)
    session = TuningSession(source, front, back, TUNABLE, reference=reference,
                            warmup=args.warmup)
    if args.params:
        try:
            path = resolve_preset(PRESETS_DIR, args.params)
        except (FileNotFoundError, ValueError) as e:
            raise SystemExit(str(e))
        session.set_values(load_params(path))
        Handler.preset, Handler.preset_values = path.stem, session.values()
        print(f"starting from preset '{path.stem}'")

    def progress(k, n):
        print(f"\rsteps 0–3: {k}/{n} frames", end="", flush=True)

    print(f"{source}\ncaching steps 0–3 …")
    dt = session.build_front(progress)
    print(f"\rsteps 0–3 cached in {dt:.1f} s ({session.n} frames)")

    Handler.session = session
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"tuner running at {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
        sys.exit(0)


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Thermal Tuner</title>
<style>
:root {
  --bg: #15171a; --panel: #1d2024; --line: #2c3036; --text: #e6e8eb;
  --muted: #9aa1aa; --accent: #e8a33d; --sel: #2b3340; --warn: #e5534b;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--text);
  font: 13px/1.4 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
header { display: flex; align-items: center; gap: 16px; padding: 10px 16px;
  border-bottom: 1px solid var(--line); background: var(--panel); }
header h1 { font-size: 15px; margin: 0; font-weight: 600; }
#preset { color: var(--accent); }
#status { color: var(--muted); flex: 1; text-align: right; }
#status.err { color: var(--warn); }
main { display: grid; grid-template-columns: 1fr 360px; min-height: calc(100vh - 45px); }
#view { padding: 12px 16px; overflow: auto; }
#side { border-left: 1px solid var(--line); background: var(--panel);
  padding: 12px 14px; overflow-y: auto; max-height: calc(100vh - 45px); }
.images { display: flex; gap: 10px; flex-wrap: wrap; align-items: flex-start; }
figure { margin: 0; }
figcaption { display: flex; justify-content: space-between; gap: 12px;
  color: var(--muted); padding: 4px 2px; font-size: 12px; }
figcaption b { color: var(--text); font-weight: 600; }
img.frame { display: block; background: #000; width: calc(640px * var(--zoom, 1)); height: auto; }
body.pixel img.frame { image-rendering: pixelated; }
.transport { display: flex; align-items: center; gap: 8px; margin: 12px 0 6px; flex-wrap: wrap; }
.transport input[type=range] { flex: 1; min-width: 200px; }
button { background: #2a2e34; color: var(--text); border: 1px solid #3a3f46;
  border-radius: 5px; padding: 5px 10px; font: inherit; cursor: pointer; }
button:hover { border-color: var(--accent); }
button.primary { background: var(--accent); color: #1a1a1a; border-color: var(--accent); font-weight: 600; }
button.warn { background: var(--warn); border-color: var(--warn); color: #fff; }
button:disabled { opacity: .4; cursor: default; }
input[type=number], input[type=text], select { background: #111; color: var(--text);
  border: 1px solid #3a3f46; border-radius: 4px; padding: 4px 6px; font: inherit; }
input[type=number] { width: 70px; }
.keys { display: flex; gap: 6px; flex-wrap: wrap; align-items: center; color: var(--muted); }
.keys button { padding: 3px 8px; }
.opts { display: flex; gap: 14px; align-items: center; color: var(--muted); margin-top: 8px; flex-wrap: wrap; }
.help-keys { margin-top: 12px; color: var(--muted); font-size: 12px; line-height: 1.9; }
kbd { background: #2a2e34; border: 1px solid #3a3f46; border-radius: 3px; padding: 0 5px; font-size: 11px; color: var(--text); }
h2 { font-size: 12px; text-transform: uppercase; letter-spacing: .06em;
  color: var(--muted); margin: 16px 0 6px; }
h2:first-child { margin-top: 0; }
.box { border: 1px solid var(--line); border-radius: 6px; padding: 10px; margin-bottom: 6px; }
.row { display: flex; gap: 6px; margin-top: 6px; }
.row > input[type=text], .row > select { flex: 1; min-width: 0; }
.param { padding: 6px 8px; border-bottom: 1px solid var(--line); border-radius: 4px; cursor: pointer; }
.param.sel { background: var(--sel); outline: 1px solid var(--accent); }
.param .top { display: flex; justify-content: space-between; gap: 8px; }
.param .val { font-variant-numeric: tabular-nums; color: var(--accent); }
.param .val.changed::after { content: " •"; color: var(--accent); }
.param input[type=range] { width: 100%; margin: 4px 0 0; }
.param .help { color: var(--muted); font-size: 11px; }
.param.toggle .top { align-items: center; }
.sw { font-size: 11px; font-weight: 700; padding: 1px 8px; border-radius: 10px;
  background: #3a3f46; color: var(--muted); min-width: 38px; text-align: center; }
.sw.on { background: var(--accent); color: #1a1a1a; }
.param.off-changed .sw { outline: 1px dashed var(--accent); }
.actions { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 14px; }
.note { color: var(--muted); font-size: 11px; margin-top: 6px; }
</style>
</head>
<body>
<header>
  <h1>Thermal Tuner</h1>
  <span>preset: <b id="preset">— defaults —</b></span>
  <span id="status">loading…</span>
</header>
<main>
  <section id="view">
    <div class="images">
      <figure><figcaption><b>Pipeline</b><span id="infoL"></span></figcaption>
        <img id="imgL" class="frame" alt="pipeline output"></figure>
      <figure id="figR"><figcaption><b>Naive (script.py)</b><span id="infoR"></span></figcaption>
        <img id="imgR" class="frame" alt="naive output"></figure>
    </div>
    <div class="transport">
      <button id="prev" title="previous frame (A)">◀</button>
      <button id="play" title="play / pause (Space)">▶ Play</button>
      <button id="next" title="next frame (D)">▶</button>
      <input id="slider" type="range" min="0" value="0" tabindex="-1">
      <input id="pos" type="number" min="0" value="0">
      <span id="ofn" class="keys"></span>
    </div>
    <div class="keys" id="keyframes">Key frames:</div>
    <div class="opts">
      <label><input type="checkbox" id="showR" checked> show naive</label>
      <label>zoom <select id="zoom"><option>1</option><option>1.5</option><option>2</option></select></label>
      <label><input type="checkbox" id="pixel"> show pixels (no smoothing)</label>
    </div>
    <div class="help-keys">
      Frames: <kbd>A</kbd> back · <kbd>D</kbd> forward · <kbd>Shift</kbd>+<kbd>A</kbd>/<kbd>D</kbd> 10 frames · <kbd>Space</kbd> play / pause<br>
      Parameters: <kbd>↑</kbd> <kbd>↓</kbd> select · <kbd>←</kbd> <kbd>→</kbd> change (switch: on / off) · <kbd>Shift</kbd>+<kbd>←</kbd>/<kbd>→</kbd> 10 steps · <kbd>R</kbd> reset selected · ↻ = needs rebuild
    </div>
  </section>
  <aside id="side">
    <h2>Presets</h2>
    <div class="box">
      <div class="row"><select id="plist"></select><button id="load">Load</button></div>
      <div class="note" id="pinfo"></div>
      <div class="row"><input type="text" id="pname" placeholder="name, e.g. dark-ground-v2"></div>
      <div class="row"><input type="text" id="pnote" placeholder="note (optional)"><button class="primary" id="save">Save</button></div>
      <div class="note" id="savenote"></div>
    </div>
    <div id="params"></div>
    <div class="actions">
      <button id="reset">Reset all to defaults</button>
      <button id="rebuild" disabled>Rebuild steps 0–3</button>
    </div>
    <div class="note">Values marked ↻ change the cached steps 0–3;
      press <b>Rebuild</b> (≈ 12 s) to see them.</div>
  </aside>
</main>
<script>
const $ = id => document.getElementById(id);
let S = null, pos = 0, playing = false, busy = false, wanted = null, sel = 0;

function setStatus(t, err) { $("status").textContent = t; $("status").className = err ? "err" : ""; }
async function api(path, body) {
  const r = await fetch(path, body === undefined ? {} :
    {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)});
  const j = await r.json();
  if (j.error) throw new Error(j.error);
  return j;
}
function fmt(p, v) { return p.integer ? String(Math.round(v)) : (+v).toFixed(p.step < 0.01 ? 3 : 2); }

// ----- presets ---------------------------------------------------------------
function showPresets() {
  const l = $("plist"); l.innerHTML = "";
  if (!S.presets.length) { const o = document.createElement("option"); o.textContent = "(no presets yet)"; o.value = ""; l.appendChild(o); }
  for (const p of S.presets) {
    const o = document.createElement("option"); o.value = p.name;
    o.textContent = p.name + (p.saved_at ? "  ·  " + p.saved_at : "");
    if (p.name === S.preset) o.selected = true;
    l.appendChild(o);
  }
  $("preset").textContent = S.preset ? S.preset + (S.preset_modified ? " (modified)" : "") : "— defaults —";
  const cur = S.presets.find(p => p.name === l.value);
  $("pinfo").textContent = cur && cur.note ? "note: " + cur.note : "";
}
$("plist").onchange = () => { const cur = S.presets.find(p => p.name === $("plist").value);
  $("pinfo").textContent = cur && cur.note ? "note: " + cur.note : ""; };
// Keep S.params: its entries hold the links to the sliders (_set, _show, _row).
// The server sends a plain copy of the parameter list, which must not replace them.
function applyState(st) {
  const {params, ...rest} = st;
  Object.assign(S, rest);
  refreshParams(); showPresets();
}

// ----- parameters ------------------------------------------------------------
function buildParams() {
  const box = $("params"); box.innerHTML = "";
  const names = {algorithms: "Algorithms (on / off)", display: "Display (instant)",
                 denoise: "Denoise (needs rebuild)",
                 michael: "From Michael's thermal_isp (M) + DDE + top-hat"};
  let lastGroup = null;
  S.params.forEach((p, idx) => {
    if (p.group !== lastGroup) {
      const h = document.createElement("h2"); h.textContent = names[p.group] || p.group;
      box.appendChild(h); lastGroup = p.group;
    }
    const d = document.createElement("div");
    p._row = d;
    if (p.toggle) {
      d.className = "param toggle";
      d.innerHTML = `<div class="top"><span>${p.label}${p.rebuild ? " ↻" : ""}</span><span class="sw"></span></div>
        <div class="help">${p.help}${p.rebuild ? " · needs rebuild" : ""}</div>`;
      const sw = d.querySelector(".sw");
      p._get = () => p._v;
      p._set = v => { p._v = !!v; };
      p._show = v => { sw.textContent = v ? "ON" : "OFF"; sw.classList.toggle("on", !!v);
        d.classList.toggle("off-changed", !!v !== !!S.defaults[p.key]); };
      d.addEventListener("mousedown", () => { select(idx, false); flip(p); });
    } else {
      d.className = "param";
      d.innerHTML = `<div class="top"><span>${p.label}${p.rebuild ? " ↻" : ""}</span><span class="val"></span></div>
        <input type="range" min="${p.min}" max="${p.max}" step="${p.step}" tabindex="-1">
        <div class="help">${p.key} — ${p.help}</div>`;
      const inp = d.querySelector("input"), val = d.querySelector(".val");
      p._get = () => +inp.value;
      p._set = v => { inp.value = v; };
      p._show = v => { val.textContent = fmt(p, v);
        val.classList.toggle("changed", Math.abs(v - S.defaults[p.key]) > 1e-9); };
      inp.addEventListener("input", () => { select(idx, false); p._show(+inp.value); queueParam(p.key, +inp.value); });
      d.addEventListener("mousedown", () => select(idx, false));
    }
    box.appendChild(d);
  });
  select(0, false);
}
function flip(p) { const v = !p._get(); p._set(v); p._show(v); queueParam(p.key, v); }
function refreshParams() {
  for (const p of S.params) { p._set(S.values[p.key]); p._show(S.values[p.key]); }
  $("rebuild").disabled = !S.front_dirty;
  $("rebuild").className = S.front_dirty ? "warn" : "";
}
function select(i, scroll = true) {
  sel = (i + S.params.length) % S.params.length;
  S.params.forEach((p, k) => p._row.classList.toggle("sel", k === sel));
  if (scroll) S.params[sel]._row.scrollIntoView({block: "nearest"});
}
function nudge(steps) {
  const p = S.params[sel];
  if (p.toggle) { flip(p); return; }
  let v = p._get() + steps * p.step;
  v = Math.min(p.max, Math.max(p.min, v));
  v = Math.round(v / p.step) * p.step;                 // stay on the step grid
  v = +v.toFixed(6);
  p._set(v); p._show(v); queueParam(p.key, v);
}
function resetSelected() {
  const p = S.params[sel], v = S.defaults[p.key];
  p._set(v); p._show(v); queueParam(p.key, v);
}

let pending = {}, ptimer = null;
function queueParam(key, v) {
  pending[key] = v; clearTimeout(ptimer);
  ptimer = setTimeout(async () => {
    const values = pending; pending = {};
    try {
      const st = await api("/api/params", {values});
      S.values = st.values; S.front_dirty = st.front_dirty;
      S.preset = st.preset; S.preset_modified = st.preset_modified; showPresets();
      $("rebuild").disabled = !S.front_dirty; $("rebuild").className = S.front_dirty ? "warn" : "";
      render(pos);
    } catch (e) { setStatus(e.message, true); }
  }, 120);
}

// ----- frames ----------------------------------------------------------------
async function render(k) {
  k = Math.max(0, Math.min(S.n - 1, k));
  if (busy) { wanted = k; return; }
  busy = true; pos = k;
  $("slider").value = k; $("pos").value = S.indices[k];
  try {
    const r = await api("/api/render?k=" + k);
    $("imgL").src = "data:image/png;base64," + r.left;
    if (r.right) $("imgR").src = "data:image/png;base64," + r.right;
    const i = r.info;
    $("infoL").textContent = `white ${i.white_pct}% · black ${i.black_pct}% · mean ${i.mean}`;
    $("infoR").textContent = r.right ? `white ${i.ref_white_pct}% · black ${i.ref_black_pct}%` : "";
    setStatus(`frame ${i.frame} · ${i.render_ms} ms` + (i.warmup_frames ? ` (warm-up ${i.warmup_frames})` : "")
      + (S.front_dirty ? " · denoise changed – press Rebuild" : ""));
  } catch (e) { setStatus(e.message, true); playing = false; updatePlay(); }
  busy = false;
  if (wanted !== null) { const w = wanted; wanted = null; render(w); return; }
  if (playing) {
    if (pos >= S.n - 1) { playing = false; updatePlay(); }
    else setTimeout(() => render(pos + 1), 0);
  }
}
function updatePlay() { $("play").textContent = playing ? "❚❚ Pause" : "▶ Play"; }
function step(d) { playing = false; updatePlay(); render(pos + d); }
function togglePlay() { playing = !playing; updatePlay(); if (playing) render(pos >= S.n - 1 ? 0 : pos + 1); }
function goFrame(f) { const k = S.indices.indexOf(f); if (k >= 0) { playing = false; updatePlay(); render(k); } }

// ----- keyboard (physical keys: works with any keyboard layout) ---------------
document.addEventListener("keydown", e => {
  const t = e.target;
  if (t.tagName === "INPUT" && (t.type === "text" || t.type === "number")) return;
  if (t.tagName === "SELECT") return;
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  const big = e.shiftKey ? 10 : 1;
  switch (e.code) {
    case "KeyA": step(-big); break;
    case "KeyD": step(big); break;
    case "Space": togglePlay(); break;
    case "ArrowUp": select(sel - 1); break;
    case "ArrowDown": select(sel + 1); break;
    case "ArrowLeft": nudge(-big); break;
    case "ArrowRight": nudge(big); break;
    case "KeyR": resetSelected(); break;
    default: return;
  }
  e.preventDefault();
}, true);

// ----- buttons -----------------------------------------------------------------
// After a mouse click, take the focus away from buttons and checkboxes, so
// Space / arrows keep controlling frames and parameters.
document.addEventListener("click", e => {
  const t = e.target; if (t.tagName === "BUTTON" || (t.tagName === "INPUT" && t.type === "checkbox")) t.blur(); });
$("prev").onclick = () => step(-1);
$("next").onclick = () => step(1);
$("play").onclick = togglePlay;
$("slider").oninput = e => { playing = false; updatePlay(); render(+e.target.value); };
$("pos").onchange = e => { goFrame(+e.target.value); e.target.blur(); };
$("showR").onchange = e => { $("figR").style.display = e.target.checked ? "" : "none"; };
$("zoom").onchange = e => { document.body.style.setProperty("--zoom", e.target.value); e.target.blur(); };
$("pixel").onchange = e => { document.body.classList.toggle("pixel", e.target.checked); e.target.blur(); };
$("save").onclick = async () => {
  const name = $("pname").value.trim();
  if (!name) { $("savenote").textContent = "Enter a name first."; $("pname").focus(); return; }
  try {
    let st = await api("/api/save", {name, note: $("pnote").value});
    if (st.exists) {
      if (!confirm(`Preset "${st.exists}" exists. Overwrite?`)) return;
      st = await api("/api/save", {name, note: $("pnote").value, overwrite: true});
    }
    applyState(st);
    $("savenote").textContent = `Saved as presets/${S.preset}.json — use: python main.py --params "${S.preset}"`;
    $("pname").blur(); $("pnote").blur();
  } catch (e) { $("savenote").textContent = e.message; }
};
$("load").onclick = async () => {
  const name = $("plist").value; if (!name) return;
  try { applyState(await api("/api/load", {name})); $("pname").value = name; render(pos);
    $("savenote").textContent = `Loaded "${name}".`; $("plist").blur(); }
  catch (e) { setStatus(e.message, true); }
};
$("reset").onclick = async () => { try { applyState(await api("/api/reset", {})); render(pos); } catch (e) { setStatus(e.message, true); } };
$("rebuild").onclick = async () => {
  $("rebuild").disabled = true; setStatus("rebuilding steps 0–3 … (≈ 12 s)");
  try { applyState(await api("/api/rebuild", {})); render(pos); } catch (e) { setStatus(e.message, true); }
};

async function init() {
  try { S = await api("/api/state"); } catch (e) { setStatus(e.message, true); return; }
  $("slider").max = S.n - 1;
  $("ofn").textContent = `of ${S.indices[S.n - 1]}`;
  for (const f of S.key_frames) if (S.indices.includes(f)) {
    const b = document.createElement("button"); b.textContent = f; b.tabIndex = -1;
    b.onclick = () => goFrame(f); $("keyframes").appendChild(b);
  }
  if (S.preset) $("pname").value = S.preset;
  buildParams(); refreshParams(); showPresets();
  render(S.indices.includes(100) ? S.indices.indexOf(100) : 0);
}
init();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
