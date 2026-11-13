"""
File: dashboard_ui.py
Project: Smart Health Monitoring System (REVA University - R24EJ806)
Author: Vasanth A N
Description: Extends the FastAPI backend with a caregiver web dashboard endpoint
             (/dashboard) and JSON query endpoints (/api/v1/alerts, /api/v1/summary)
             for real-time patient monitoring, sensor quality inspection, and alert
             auditing.
"""
import json
import sqlite3
from typing import List

import numpy as np
from fastapi import APIRouter, Depends, Query
from fastapi.responses import HTMLResponse

from settings import DB_PATH, DEFAULT_DEVICE_ID, require_caregiver_token

router = APIRouter()

DEVICE_ID_PATTERN = r"^[A-Za-z0-9_\-]{1,64}$"


def _query(sql: str, params: tuple) -> List[dict]:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = [dict(r) for r in conn.execute(sql, params).fetchall()]
    for r in rows:
        r["explanation_codes"] = json.loads(r.get("explanation_codes") or "[]")
    return rows


@router.get("/api/v1/summary", dependencies=[Depends(require_caregiver_token)])
def get_system_summary(
    device_id: str = Query(default=DEFAULT_DEVICE_ID, pattern=DEVICE_ID_PATTERN),
    limit: int = Query(default=25, ge=1, le=500),
) -> dict:
    """Returns latest vital signs, hybrid score components, and latency stats for a device."""
    rows = _query(
        "SELECT * FROM telemetry_log WHERE device_id = ? ORDER BY id DESC LIMIT ?",
        (device_id, limit),
    )
    if not rows:
        return {"device_id": device_id, "status": "NO_DATA", "history": []}

    # Replayed (buffered) packets measure outage duration, not network latency
    live = [r["latency_ms"] for r in rows if not r["replayed"]]
    alerts = [r for r in rows if r["alert_state"] in ("WATCH", "CRITICAL")]
    return {
        "device_id": device_id,
        "status": "OK",
        "latest_record": rows[0],
        "mean_latency_ms": round(float(np.mean(live)), 2) if live else None,
        "p95_latency_ms": round(float(np.percentile(live, 95)), 2) if live else None,
        "recent_alert_count": len(alerts),
        "history": rows,
    }


@router.get("/api/v1/alerts", dependencies=[Depends(require_caregiver_token)])
def get_alert_log(
    device_id: str = Query(default=DEFAULT_DEVICE_ID, pattern=DEVICE_ID_PATTERN),
    limit: int = Query(default=50, ge=1, le=500),
) -> dict:
    """Audit log of WATCH/CRITICAL samples with their explanation codes."""
    rows = _query(
        """
        SELECT id, seq_id, t_acquire_ms, alert_state, r_score, m_score, q_score, a_score,
               quality_flags, explanation_codes
        FROM telemetry_log
        WHERE device_id = ? AND alert_state IN ('WATCH', 'CRITICAL')
        ORDER BY id DESC LIMIT ?
        """,
        (device_id, limit),
    )
    return {"device_id": device_id, "alerts": rows}


@router.get("/dashboard", response_class=HTMLResponse)
def render_caregiver_dashboard() -> str:
    """Static page shell. It holds no patient data: the page asks for the caregiver
    token and fetches data from the authenticated JSON endpoints above."""
    return DASHBOARD_HTML


DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Caregiver Dashboard</title>
<style>
  body { font-family: system-ui, sans-serif; margin: 0; background: #f4f6f9; color: #1d2733; }
  header { background: #1f3a5f; color: #fff; padding: 12px 16px; }
  main { max-width: 1100px; margin: 0 auto; padding: 16px; }
  .row { display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 16px; }
  .tile { background: #fff; border-radius: 8px; padding: 12px 16px; min-width: 150px; flex: 1;
          box-shadow: 0 1px 2px rgba(0,0,0,.08); }
  .tile .label { font-size: 12px; color: #5a6b7d; }
  .tile .value { font-size: 24px; font-weight: 600; }
  .NORMAL { color: #1b7f3b; } .WATCH { color: #b26a00; } .CRITICAL { color: #b3261e; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 12px;
           margin: 2px; background: #e7ecf2; }
  .badge.bad { background: #fde2e1; color: #b3261e; }
  table { width: 100%; border-collapse: collapse; background: #fff; font-size: 13px; }
  th, td { padding: 6px 8px; border-bottom: 1px solid #e3e8ee; text-align: left; }
  #login { background: #fff; padding: 16px; border-radius: 8px; }
  input { padding: 6px; width: 260px; }
  .scroll { overflow-x: auto; }
</style>
</head>
<body>
<header><strong>Smart Health Monitoring</strong> &mdash; Caregiver Dashboard (monitoring aid, not a diagnosis)</header>
<main>
  <div id="login">
    <label>Device ID <input id="device" value="DEV_ESP32_NODE_03"></label><br><br>
    <label>Caregiver token <input id="token" type="password"></label>
    <button id="go">Open</button> <span id="err" class="CRITICAL"></span>
  </div>
  <div id="app" hidden>
    <div class="row" id="vitals"></div>
    <div class="row" id="scores"></div>
    <div class="row"><div class="tile"><div class="label">Sensor quality</div><div id="quality"></div></div></div>
    <h3>Alert log</h3>
    <div class="scroll"><table><thead><tr>
      <th>Seq</th><th>Acquired</th><th>State</th><th>R_t</th><th>M_t</th><th>Q_t</th><th>A_t</th><th>Codes</th>
    </tr></thead><tbody id="alerts"></tbody></table></div>
  </div>
</main>
<script>
const FLAGS = {1: "PPG detached", 2: "HR implausible", 4: "SpO2 implausible",
               8: "Temp implausible", 16: "BP implausible", 32: "Motion artifact"};
let token = sessionStorage.getItem("cg_token") || "";

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;   // textContent: no HTML injection
  return e;
}
function tile(label, value, cls) {
  const t = el("div", "tile");
  t.append(el("div", "label", label), el("div", "value " + (cls || ""), value));
  return t;
}
const fmt = (v, d) => (v === null || v === undefined) ? "--" : Number(v).toFixed(d);

async function api(path) {
  const dev = encodeURIComponent(document.getElementById("device").value);
  const r = await fetch(path + "?device_id=" + dev, {headers: {Authorization: "Bearer " + token}});
  if (r.status === 401) throw new Error("Token rejected");
  if (!r.ok) throw new Error("HTTP " + r.status);
  return r.json();
}

async function refresh() {
  try {
    const s = await api("/api/v1/summary");
    const a = await api("/api/v1/alerts");
    document.getElementById("login").hidden = true;
    document.getElementById("app").hidden = false;
    const v = document.getElementById("vitals"), sc = document.getElementById("scores");
    v.replaceChildren(); sc.replaceChildren();
    if (s.status === "NO_DATA") { v.append(tile("Status", "No data yet")); return; }
    const L = s.latest_record;
    v.append(tile("Heart rate (bpm)", fmt(L.hr, 0)), tile("SpO2 (%)", fmt(L.spo2, 1)),
             tile("Temp (C)", fmt(L.temp, 1)),
             tile("BP (mmHg)", fmt(L.bp_sys, 0) + " / " + fmt(L.bp_dia, 0)),
             tile("Alert state", L.alert_state, L.alert_state));
    sc.append(tile("R_t rule", fmt(L.r_score, 2)), tile("M_t ML", fmt(L.m_score, 2)),
              tile("Q_t quality", fmt(L.q_score, 2)), tile("A_t hybrid", fmt(L.a_score, 2)),
              tile("Mean latency (ms)", fmt(s.mean_latency_ms, 0)),
              tile("P95 latency (ms)", fmt(s.p95_latency_ms, 0)));
    const q = document.getElementById("quality"); q.replaceChildren();
    const bad = Object.keys(FLAGS).filter(b => L.quality_flags & b);
    if (!bad.length) q.append(el("span", "badge", "All sensors valid"));
    bad.forEach(b => q.append(el("span", "badge bad", FLAGS[b])));
    const tb = document.getElementById("alerts"); tb.replaceChildren();
    a.alerts.forEach(r => {
      const tr = el("tr");
      tr.append(el("td", "", r.seq_id), el("td", "", new Date(r.t_acquire_ms).toLocaleString()),
                el("td", r.alert_state, r.alert_state), el("td", "", fmt(r.r_score, 2)),
                el("td", "", fmt(r.m_score, 2)), el("td", "", fmt(r.q_score, 2)),
                el("td", "", fmt(r.a_score, 2)), el("td", "", r.explanation_codes.join(", ")));
      tb.append(tr);
    });
  } catch (e) {
    document.getElementById("err").textContent = e.message;
    document.getElementById("login").hidden = false;
  }
}

document.getElementById("go").onclick = () => {
  token = document.getElementById("token").value;
  sessionStorage.setItem("cg_token", token);
  refresh();
};
if (token) refresh();
setInterval(() => { if (token) refresh(); }, 5000);
</script>
</body>
</html>
"""
