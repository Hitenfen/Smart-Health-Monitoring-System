"""HealthSync caregiver dashboard and patient testing portal.

Keeps the existing FastAPI + SQLite API structure and adds a browser portal.
Test readings submitted through the portal are clearly marked as simulated.
"""
import json
import sqlite3
from typing import List

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import HTMLResponse

from settings import DB_PATH, DEFAULT_DEVICE_ID, require_caregiver_token

router = APIRouter()
DEVICE_ID_PATTERN = r"^[A-Za-z0-9_\-]{1,64}$"


def _query(sql: str, params: tuple) -> List[dict]:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = [dict(r) for r in conn.execute(sql, params).fetchall()]
    for row in rows:
        try:
            row["explanation_codes"] = json.loads(row.get("explanation_codes") or "[]")
        except (TypeError, json.JSONDecodeError):
            row["explanation_codes"] = []
    return rows


@router.get("/api/v1/devices", dependencies=[Depends(require_caregiver_token)])
def get_devices() -> dict:
    """Return device IDs that actually exist in the local telemetry database."""
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT device_id, COUNT(*) AS record_count, MAX(id) AS latest_id "
            "FROM telemetry_log GROUP BY device_id ORDER BY latest_id DESC"
        ).fetchall()
    return {"devices": [{"device_id": r[0], "record_count": r[1]} for r in rows]}


@router.get("/api/v1/summary", dependencies=[Depends(require_caregiver_token)])
def get_system_summary(
    device_id: str = Query(default=DEFAULT_DEVICE_ID, pattern=DEVICE_ID_PATTERN),
    limit: int = Query(default=25, ge=1, le=500),
) -> dict:
    rows = _query(
        "SELECT * FROM telemetry_log WHERE device_id = ? ORDER BY id DESC LIMIT ?",
        (device_id, limit),
    )
    if not rows:
        return {"device_id": device_id, "status": "NO_DATA", "history": []}
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
    rows = _query(
        "SELECT id, seq_id, t_acquire_ms, alert_state, r_score, m_score, q_score, a_score, "
        "quality_flags, explanation_codes FROM telemetry_log "
        "WHERE device_id = ? AND alert_state IN ('WATCH', 'CRITICAL') "
        "ORDER BY id DESC LIMIT ?",
        (device_id, limit),
    )
    return {"device_id": device_id, "alerts": rows}


@router.get("/portal", response_class=HTMLResponse)
@router.get("/dashboard", response_class=HTMLResponse)
def render_portal() -> str:
    return DASHBOARD_HTML


DASHBOARD_HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>HealthSync | Smart Health Monitoring</title>
<style>
:root{--ink:#17243a;--muted:#718096;--line:#e6ebf2;--bg:#f5f7fb;--blue:#3157d5;--teal:#0d9488;--red:#dc5252;--amber:#d18b22;--card:#fff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 Inter,system-ui,Segoe UI,sans-serif}
header{background:#172b4d;color:#fff;padding:17px 25px;display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap}
.brand{font-size:20px;font-weight:750;letter-spacing:.2px}.brand span{color:#73d6cc}.sub{color:#c5d1e3;font-size:12px}
main{max-width:1240px;margin:auto;padding:24px}.login,.panel,.metric{background:var(--card);border:1px solid var(--line);border-radius:15px;box-shadow:0 5px 20px #1c31500a}
.login{max-width:520px;margin:55px auto;padding:25px}.panel{padding:19px;margin-bottom:18px}.topline{display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap;margin-bottom:15px}
h1{font-size:25px;margin:0 0 4px}h2{font-size:16px;margin:0 0 14px}.muted{color:var(--muted)}label{display:block;font-weight:650;font-size:12px;color:#526176;margin:0 0 6px}
input,select{width:100%;padding:11px 12px;border:1px solid #d9e1ec;border-radius:9px;background:#fff;color:var(--ink);font:inherit;outline:none}input:focus,select:focus{border-color:var(--blue)}
.field{margin-bottom:13px}.grid{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:13px}.metric{padding:16px}.metric .k{font-size:12px;color:var(--muted)}.metric .v{font-size:25px;font-weight:750;margin-top:5px;letter-spacing:-.5px}.unit{font-size:12px;color:var(--muted);font-weight:500}
.btn{border:0;border-radius:9px;padding:10px 15px;background:var(--blue);color:white;font-weight:700;cursor:pointer}.btn.secondary{background:#eaf0ff;color:#2948ac}.btn.dark{background:#233653}.btn:disabled{opacity:.55;cursor:not-allowed}.actions{display:flex;gap:9px;flex-wrap:wrap}.status{font-weight:800}.NORMAL{color:#12805c}.WATCH{color:#b7791f}.CRITICAL{color:#c53030}.pill{display:inline-block;border-radius:99px;padding:4px 9px;background:#edf2f7;font-size:11px;margin:2px}.bad{background:#fff0ef;color:#b42318}.good{background:#e7f8f1;color:#087451}
.tabs{display:flex;gap:8px;flex-wrap:wrap;margin:18px 0}.tab{border:1px solid var(--line);background:#fff;color:#526176;border-radius:9px;padding:9px 13px;cursor:pointer;font-weight:650}.tab.active{background:#eaf0ff;color:#2948ac;border-color:#cdd9ff}.section[hidden]{display:none}.tablewrap{overflow:auto}table{width:100%;border-collapse:collapse;min-width:720px;font-size:12px}th,td{text-align:left;padding:10px;border-bottom:1px solid var(--line);white-space:nowrap}th{color:#64748b;background:#f8fafc}
.formgrid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:13px}.notice{padding:11px 13px;background:#fff8e8;color:#8b6118;border:1px solid #f5e4bb;border-radius:10px;font-size:12px;margin-bottom:14px}.error{color:#c53030;font-size:12px;margin-top:9px}.success{color:#087451;font-size:12px;margin-top:9px}.chart{width:100%;height:220px}.legend{display:flex;gap:12px;color:var(--muted);font-size:12px;flex-wrap:wrap}.dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:#3157d5;margin-right:5px}
@media(max-width:900px){.grid{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:600px){main{padding:14px}.grid,.formgrid{grid-template-columns:1fr 1fr}.metric .v{font-size:21px}header{padding:14px}}
</style></head><body>
<header><div><div class="brand">Health<span>Sync</span></div><div class="sub">SMART HEALTH MONITORING PLATFORM</div></div><div class="sub">Caregiver workspace · Monitoring aid, not a diagnosis</div></header>
<main>
<div id="login" class="login"><h1>Welcome to HealthSync</h1><p class="muted">Sign in with your caregiver token to view stored health records.</p>
<div class="field"><label for="token">CAREGIVER TOKEN</label><input id="token" type="password" placeholder="Enter caregiver token" autocomplete="current-password"></div>
<button class="btn" id="loginBtn">Open portal</button><div id="loginError" class="error"></div></div>
<div id="workspace" hidden>
<div class="topline"><div><h1>Health overview</h1><div class="muted">Actual records retrieved from your SQLite database</div></div><div class="actions"><button class="btn secondary" id="refreshBtn">↻ Refresh</button><button class="btn dark" id="logoutBtn">Sign out</button></div></div>
<div class="panel"><div class="topline"><div><b>Monitoring device</b><div class="muted">Choose a device ID found in your database</div></div><span class="pill good" id="dbStatus">Database connected</span></div><div style="max-width:520px"><select id="deviceSelect"></select></div></div>
<div class="tabs"><button class="tab active" data-tab="overview">Overview</button><button class="tab" data-tab="patient">Patient input</button><button class="tab" data-tab="lab">Testing lab</button><button class="tab" data-tab="history">Records & alerts</button></div>
<section id="overview" class="section"><div class="grid" id="metrics"></div><div class="panel" style="margin-top:18px"><div class="topline"><div><h2>Vital readings history</h2><div class="muted">Recent stored readings · missing or invalid values are shown as gaps</div></div><div class="legend"><span><i class="dot"></i>Heart rate</span></div></div><canvas id="trend" class="chart"></canvas></div><div class="panel"><h2>Sensor quality</h2><div id="quality" class="muted">Select a device to inspect its latest reading.</div></div></section>
<section id="patient" class="section" hidden><div class="panel"><h2>Enter a test reading</h2><div class="notice">Demo mode: readings submitted here are simulated test data, not measurements from physical sensors. Blood pressure is entered manually.</div>
<div class="formgrid"><div class="field"><label>Heart rate (BPM)</label><input id="hr" type="number" min="1" max="300" placeholder="e.g. 78"></div><div class="field"><label>SpO₂ (%)</label><input id="spo2" type="number" min="1" max="100" step="0.1" placeholder="e.g. 98"></div><div class="field"><label>Temperature (°C)</label><input id="temp" type="number" min="1" max="50" step="0.1" placeholder="e.g. 36.8"></div><div class="field"><label>Systolic BP (mmHg)</label><input id="sys" type="number" min="1" max="300" placeholder="e.g. 120"></div><div class="field"><label>Diastolic BP (mmHg)</label><input id="dia" type="number" min="1" max="200" placeholder="e.g. 80"></div></div>
<div class="actions"><button class="btn" id="submitReading">Save simulated reading</button><button class="btn secondary" id="clearForm">Clear</button></div><div id="submitMessage"></div></div></section>
<section id="lab" class="section" hidden><div class="panel"><h2>Testing laboratory</h2><p class="muted">Choose a predefined scenario. Each test is saved to SQLite and processed by the existing backend pipeline.</p><div class="grid" style="grid-template-columns:repeat(3,minmax(0,1fr))"><div class="metric"><b>Normal reading</b><p class="muted">Typical sample values with no quality flags.</p><button class="btn secondary scenario" data-case="normal">Run normal test</button></div><div class="metric"><b>Boundary reading</b><p class="muted">HR 221 BPM with the HR-implausible flag.</p><button class="btn secondary scenario" data-case="boundary">Run boundary test</button></div><div class="metric"><b>Abnormal reading</b><p class="muted">Low oxygen sample to exercise rule-based monitoring.</p><button class="btn secondary scenario" data-case="abnormal">Run abnormal test</button></div></div><div id="labMessage"></div></div></section>
<section id="history" class="section" hidden><div class="panel"><h2>Recent telemetry records</h2><div class="tablewrap"><table><thead><tr><th>Sequence</th><th>Time</th><th>HR</th><th>SpO₂</th><th>Temp</th><th>BP</th><th>Quality flags</th><th>Alert</th></tr></thead><tbody id="records"></tbody></table></div></div><div class="panel"><h2>Alert log</h2><div class="tablewrap"><table><thead><tr><th>Sequence</th><th>Time</th><th>State</th><th>Rule</th><th>ML</th><th>Quality</th><th>Hybrid</th><th>Codes</th></tr></thead><tbody id="alerts"></tbody></table></div></div></section>
<div class="muted" style="font-size:11px;text-align:center;padding:10px">HealthSync prototype · For monitoring and academic demonstration only; not a diagnostic device.</div>
</div></main>
<script>
const $=id=>document.getElementById(id);let token=sessionStorage.getItem('cg_token')||'';let currentSummary=null;let refreshTimer=null;
const FLAGS={1:'PPG detached',2:'HR implausible',4:'SpO₂ implausible',8:'Temperature implausible',16:'BP implausible',32:'Motion artifact'};
const fmt=(v,d=1)=>v===null||v===undefined||Number.isNaN(Number(v))?'—':Number(v).toFixed(d);
function authHeaders(){return {'Authorization':'Bearer '+token,'Content-Type':'application/json'}}
async function api(path,opts={}){const r=await fetch(path,{...opts,headers:{...authHeaders(),...(opts.headers||{})}});if(r.status===401)throw Error('Caregiver token was rejected. Please sign in again.');if(!r.ok){let m='HTTP '+r.status;try{m=(await r.json()).detail||m}catch{}throw Error(m)}return r.json()}
function showMessage(id,msg,ok=true){$(id).className=ok?'success':'error';$(id).textContent=msg}
function metric(label,value,unit=''){return `<div class="metric"><div class="k">${label}</div><div class="v">${value}<span class="unit"> ${unit}</span></div></div>`}
function activateTab(name){document.querySelectorAll('.tab').forEach(b=>b.classList.toggle('active',b.dataset.tab===name));document.querySelectorAll('.section').forEach(s=>s.hidden=s.id!==name)}
document.querySelectorAll('.tab').forEach(b=>b.addEventListener('click',()=>activateTab(b.dataset.tab)));
async function loadDevices(){const data=await api('/api/v1/devices');const sel=$('deviceSelect');const previous=sel.value;sel.replaceChildren();data.devices.forEach((d,index)=>{const o=document.createElement('option');o.value=d.device_id;o.textContent=`Patient ${index+1} · ${d.record_count} record(s)`;o.title=d.device_id;sel.append(o)});if(previous&&[...sel.options].some(o=>o.value===previous))sel.value=previous;if(!data.devices.length){const o=document.createElement('option');o.textContent='No patients found';sel.append(o)}}
function drawTrend(rows){const c=$('trend'),ctx=c.getContext('2d'),w=c.clientWidth||700,h=220,dpr=window.devicePixelRatio||1;c.width=w*dpr;c.height=h*dpr;ctx.scale(dpr,dpr);ctx.clearRect(0,0,w,h);const vals=rows.slice().reverse().map(r=>r.hr===null?null:Number(r.hr));const valid=vals.filter(v=>v!==null&&v>0&&v<=220);if(valid.length<1){ctx.fillStyle='#718096';ctx.font='13px system-ui';ctx.fillText('No valid heart-rate values available for this device.',12,30);return}const min=Math.min(40,...valid)-5,max=Math.max(120,...valid)+5;ctx.strokeStyle='#e6ebf2';ctx.lineWidth=1;for(let i=0;i<4;i++){let y=15+i*(h-40)/3;ctx.beginPath();ctx.moveTo(35,y);ctx.lineTo(w-12,y);ctx.stroke()}ctx.fillStyle='#718096';ctx.font='10px system-ui';ctx.fillText(String(Math.round(max)),4,20);ctx.fillText(String(Math.round(min)),4,h-25);ctx.strokeStyle='#3157d5';ctx.lineWidth=2.5;ctx.beginPath();let started=false;vals.forEach((v,i)=>{if(v===null||v<=0||v>220){started=false;return}const x=38+i*(w-55)/Math.max(1,vals.length-1),y=15+(max-v)*(h-40)/(max-min);if(!started){ctx.moveTo(x,y);started=true}else ctx.lineTo(x,y)});ctx.stroke()}
function renderSummary(s,a){currentSummary=s;const host=$('metrics');host.replaceChildren();if(s.status==='NO_DATA'){host.innerHTML=metric('Status','No records');$('quality').textContent='No readings are stored for this device.';$('records').replaceChildren();$('alerts').replaceChildren();drawTrend([]);return}const r=s.latest_record;host.innerHTML=metric('Heart rate',fmt(r.hr,0),'BPM')+metric('SpO₂',fmt(r.spo2,1),'%')+metric('Temperature',fmt(r.temp,1),'°C')+metric('Blood pressure',fmt(r.bp_sys,0)+' / '+fmt(r.bp_dia,0),'mmHg')+metric('Alert state',`<span class="${r.alert_state}">${r.alert_state}</span>`)+metric('Rule score',fmt(r.r_score,2))+metric('ML score',fmt(r.m_score,2))+metric('Quality score',fmt(r.q_score,2))+metric('Hybrid score',fmt(r.a_score,2))+metric('Mean latency',fmt(s.mean_latency_ms,0),'ms');
const bad=Object.keys(FLAGS).filter(b=>(Number(r.quality_flags)&Number(b))!==0);$('quality').innerHTML=bad.length?bad.map(b=>`<span class="pill bad">${FLAGS[b]}</span>`).join(''):'<span class="pill good">No quality flags on latest record</span>';
$('records').innerHTML=s.history.map(x=>`<tr><td>${x.seq_id}</td><td>${new Date(x.t_acquire_ms).toLocaleString()}</td><td>${fmt(x.hr,0)}</td><td>${fmt(x.spo2,1)}</td><td>${fmt(x.temp,1)}</td><td>${fmt(x.bp_sys,0)} / ${fmt(x.bp_dia,0)}</td><td>${x.quality_flags}</td><td class="${x.alert_state}">${x.alert_state}</td></tr>`).join('');
$('alerts').innerHTML=a.alerts.length?a.alerts.map(x=>`<tr><td>${x.seq_id}</td><td>${new Date(x.t_acquire_ms).toLocaleString()}</td><td class="${x.alert_state}">${x.alert_state}</td><td>${fmt(x.r_score,2)}</td><td>${fmt(x.m_score,2)}</td><td>${fmt(x.q_score,2)}</td><td>${fmt(x.a_score,2)}</td><td>${(x.explanation_codes||[]).join(', ')}</td></tr>`).join(''):'<tr><td colspan="8">No WATCH or CRITICAL alerts for this device.</td></tr>';
drawTrend(s.history)}
async function refresh(){try{if(!$('deviceSelect').value)await loadDevices();const dev=encodeURIComponent($('deviceSelect').value);const [s,a]=await Promise.all([api('/api/v1/summary?device_id='+dev+'&limit=50'),api('/api/v1/alerts?device_id='+dev+'&limit=50')]);renderSummary(s,a);$('dbStatus').textContent='Database connected · refreshed '+new Date().toLocaleTimeString()}catch(e){$('dbStatus').textContent='Connection issue';showMessage('loginError',e.message,false);if(e.message.includes('token'))logout()}}
async function login(){token=$('token').value.trim();if(!token){showMessage('loginError','Enter your caregiver token.',false);return}sessionStorage.setItem('cg_token',token);try{await loadDevices();$('login').hidden=true;$('workspace').hidden=false;await refresh();if(refreshTimer)clearInterval(refreshTimer);refreshTimer=setInterval(refresh,10000)}catch(e){showMessage('loginError',e.message,false);sessionStorage.removeItem('cg_token');token=''}}
function logout(){token='';sessionStorage.removeItem('cg_token');if(refreshTimer)clearInterval(refreshTimer);$('workspace').hidden=true;$('login').hidden=false}
$('loginBtn').onclick=login;$('token').addEventListener('keydown',e=>{if(e.key==='Enter')login()});$('logoutBtn').onclick=logout;$('refreshBtn').onclick=refresh;$('deviceSelect').onchange=refresh;
function clearForm(){['hr','spo2','temp','sys','dia'].forEach(id=>$(id).value='');$('submitMessage').textContent=''}$('clearForm').onclick=clearForm;
async function sendReading(values,target){try{const result=await api('/api/v1/portal/test-reading',{method:'POST',body:JSON.stringify({device_id:$('deviceSelect').value,...values})});showMessage(target,`Saved simulated reading · sequence ${result.seq_id} · alert ${result.alert_state}`,true);await loadDevices();await refresh()}catch(e){showMessage(target,e.message,false)}}
$('submitReading').onclick=()=>{const ids=['hr','spo2','temp','sys','dia'];if(ids.some(id=>$(id).value.trim()==='')){showMessage('submitMessage','Please enter all five readings. Use realistic values for missing sensors only if they were actually measured.',false);return}const [hr,spo2,temp,bp_sys,bp_dia]=ids.map(id=>Number($(id).value));if(![hr,spo2,temp,bp_sys,bp_dia].every(Number.isFinite)){showMessage('submitMessage','Enter valid numeric readings.',false);return}sendReading({hr,spo2,temp,bp_sys,bp_dia,quality_flags:0},'submitMessage')};
document.querySelectorAll('.scenario').forEach(b=>b.onclick=()=>{const cases={normal:{hr:78,spo2:98,temp:36.8,bp_sys:120,bp_dia:80,quality_flags:0},boundary:{hr:221,spo2:98,temp:36.8,bp_sys:120,bp_dia:80,quality_flags:2},abnormal:{hr:105,spo2:88,temp:37.2,bp_sys:135,bp_dia:88,quality_flags:0}};sendReading(cases[b.dataset.case],'labMessage')});
if(token){$('token').value=token;login()}
</script></body></html>'''
