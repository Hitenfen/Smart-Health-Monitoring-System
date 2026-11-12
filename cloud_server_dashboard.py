"""
File: cloud_server_dashboard.py
Project: Smart Health Monitoring System (REVA University - R24EJ806)
Author: Vasanth A N
Description: Authenticated REST API server for receiving ESP32 telemetry packets,
             logging end-to-end latency L_i, running hybrid ML anomaly detection,
             and persisting audit-ready alert histories to SQLite.

Run:  set HEALTH_DEVICE_TOKEN and HEALTH_CAREGIVER_TOKEN, then
      uvicorn cloud_server_dashboard:app --host 0.0.0.0 --port 8443 \
              --ssl-keyfile key.pem --ssl-certfile cert.pem
"""
import json
import logging
import sqlite3
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Deque, Dict, Optional

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

import dashboard_ui
from ml_hybrid_pipeline import QUAL_ALL, AlertStateMachine, HybridAnomalyDetector
from settings import DB_PATH, MODEL_PATH, check_tokens_configured, require_device_token, require_caregiver_token

log = logging.getLogger("health_cloud")

CLOCK_SKEW_TOLERANCE_MS = 2_000  # negative latency beyond this means device clock is wrong


def load_detector() -> HybridAnomalyDetector:
    """Load the model trained by run_experiments_and_plots.py; fall back to rules only."""
    path = Path(MODEL_PATH)
    if path.exists():
        detector = HybridAnomalyDetector.load(path)
        log.info("Loaded %s detector from %s", detector.mode, path)
        return detector
    log.warning("No trained model at %s: M_t will be 0 (rules + quality only)", path)
    return HybridAnomalyDetector(mode="unsupervised_iforest")


class TelemetryIn(BaseModel):
    device_id: str = Field(..., pattern=r"^[A-Za-z0-9_\-]{1,64}$",
                           description="Pseudonymous device identifier")
    fw_ver: str = Field(..., max_length=32)
    seq_id: int = Field(..., ge=0)
    t_acquire_ms: int = Field(..., gt=0, description="Unix epoch ms from the NTP-synced device clock")
    # None = invalid/missing sample (the firmware sends JSON null instead of a bad value)
    hr: Optional[float] = None
    spo2: Optional[float] = None
    temp: Optional[float] = None
    bp_sys: Optional[float] = None
    bp_dia: Optional[float] = None
    quality_flags: int = Field(..., ge=0, le=QUAL_ALL)
    replayed: bool = False  # True when sent from the store-and-forward buffer


def init_db() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS telemetry_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                device_id TEXT NOT NULL,
                fw_ver TEXT NOT NULL,
                seq_id INTEGER NOT NULL,
                t_acquire_ms INTEGER NOT NULL,
                t_receive_ms INTEGER NOT NULL,
                latency_ms INTEGER NOT NULL,
                replayed INTEGER NOT NULL DEFAULT 0,
                hr REAL, spo2 REAL, temp REAL, bp_sys REAL, bp_dia REAL,
                quality_flags INTEGER NOT NULL,
                r_score REAL, m_score REAL, q_score REAL, a_score REAL,
                alert_state TEXT NOT NULL,
                explanation_codes TEXT NOT NULL,
                UNIQUE (device_id, seq_id)
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_telemetry_device ON telemetry_log (device_id, id)"
        )


@asynccontextmanager
async def lifespan(_: FastAPI):
    check_tokens_configured()
    init_db()
    yield


app = FastAPI(title="RIO Smart Health Monitoring Cloud Service", version="1.2.1", lifespan=lifespan)
app.include_router(dashboard_ui.router)

detector = load_detector()
_history: Dict[str, Deque[dict]] = {}
_alert_machines: Dict[str, AlertStateMachine] = {}
_state_lock = threading.Lock()  # sync endpoints run in a thread pool


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": detector.is_fitted}


@app.post("/api/v1/telemetry", dependencies=[Depends(require_device_token)])
def ingest_telemetry(pkt: TelemetryIn) -> dict:
    t_receive_ms = time.time_ns() // 1_000_000
    latency_ms = t_receive_ms - pkt.t_acquire_ms  # L_i = t_receive - t_acquire

    with _state_lock, sqlite3.connect(DB_PATH) as conn:
        # Store-and-forward retries can resend a packet; acknowledge it without re-scoring
        dup = conn.execute(
            "SELECT alert_state FROM telemetry_log WHERE device_id = ? AND seq_id = ?",
            (pkt.device_id, pkt.seq_id),
        ).fetchone()
        if dup:
            return {"status": "duplicate", "seq_id": pkt.seq_id, "alert_state": dup[0]}

        # Rolling features need the last N samples of this device, not a single packet
        history = _history.setdefault(
            pkt.device_id, deque(maxlen=detector.preprocessor.window_n)
        )
        history.append(pkt.model_dump())
        window = pd.DataFrame(list(history))
        window[detector.group_col] = pkt.device_id
        latest = detector.score(window).iloc[-1]

        machine = _alert_machines.setdefault(pkt.device_id, AlertStateMachine(detector.weights))
        alert_state, codes = machine.update(
            float(latest["A_t"]), pkt.quality_flags, latest["rule_codes"], float(latest["R_t"])
        )
        if latency_ms < -CLOCK_SKEW_TOLERANCE_MS:
            codes.append("DEVICE_CLOCK_SKEW")

        conn.execute(
            """
            INSERT INTO telemetry_log (
                device_id, fw_ver, seq_id, t_acquire_ms, t_receive_ms, latency_ms, replayed,
                hr, spo2, temp, bp_sys, bp_dia, quality_flags,
                r_score, m_score, q_score, a_score, alert_state, explanation_codes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                pkt.device_id, pkt.fw_ver, pkt.seq_id, pkt.t_acquire_ms, t_receive_ms,
                latency_ms, int(pkt.replayed),
                pkt.hr, pkt.spo2, pkt.temp, pkt.bp_sys, pkt.bp_dia, pkt.quality_flags,
                float(latest["R_t"]), float(latest["M_t"]), float(latest["Q_t"]),
                float(latest["A_t"]), alert_state, json.dumps(codes),
            ),
        )

    if alert_state != "NORMAL":
        log.warning("ALERT %s device=%s seq=%d A_t=%.3f codes=%s",
                    alert_state, pkt.device_id, pkt.seq_id, latest["A_t"], codes)
    return {
        "status": "stored",
        "seq_id": pkt.seq_id,
        "latency_ms": latency_ms,
        "a_score": round(float(latest["A_t"]), 4),
        "alert_state": alert_state,
        "explanation_codes": codes,
    }


@app.post("/api/v1/portal/test-reading", dependencies=[Depends(require_caregiver_token)])
def portal_test_reading(payload: dict) -> dict:
    """Submit a clearly labelled simulated reading from the caregiver portal.

    This endpoint is intended for the localhost academic demo. Physical devices
    must continue using POST /api/v1/telemetry with the device token.
    """
    import re
    from datetime import datetime, timezone

    device_id = str(payload.get("device_id", "")).strip()
    if not re.fullmatch(r"[A-Za-z0-9_\-]{1,64}", device_id):
        raise HTTPException(status_code=422, detail="Invalid device ID")

    allowed = {"hr", "spo2", "temp", "bp_sys", "bp_dia", "quality_flags"}
    values = {key: payload.get(key) for key in allowed}
    if any(values.get(key) is None for key in ("hr", "spo2", "temp", "bp_sys", "bp_dia")):
        raise HTTPException(status_code=422, detail="All five vital readings are required")

    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute("SELECT COALESCE(MAX(seq_id), 0) + 1 FROM telemetry_log WHERE device_id = ?", (device_id,)).fetchone()
    seq_id = int(row[0])
    packet = TelemetryIn(
        device_id=device_id, fw_ver="healthsync-portal-demo", seq_id=seq_id,
        t_acquire_ms=time.time_ns() // 1_000_000, replayed=False, **values
    )
    result = ingest_telemetry(packet)
    result["data_source"] = "SIMULATED_PORTAL_TEST"
    return result
