"""Phase 6: Cloud API (FastAPI) with REST + WebSocket skeleton."""
from __future__ import annotations

from datetime import datetime
from typing import List

from fastapi import FastAPI, WebSocket
from pydantic import BaseModel

app = FastAPI(title="Respiratory Monitoring Cloud")

ALERTS = []
DEVICES = {}
MODELS = [{"id": "tinyml-cnn-int8-v1", "created_at": datetime.utcnow().isoformat(), "acc": 0.928}]


class AlertIn(BaseModel):
    device_id: str
    timestamp: float
    confidence: float
    cls: str
    reliability_score: float


@app.post("/api/alerts")
def post_alert(alert: AlertIn):
    ALERTS.append(alert.model_dump())
    return {"status": "accepted", "count": len(ALERTS)}


@app.get("/api/alerts")
def get_alerts() -> List[dict]:
    return ALERTS[-500:]


@app.get("/api/models")
def get_models():
    return MODELS


@app.get("/api/devices")
def get_devices():
    return DEVICES


@app.get("/api/analytics")
def analytics():
    severe = sum(1 for a in ALERTS if a["cls"] == "severe")
    mild = sum(1 for a in ALERTS if a["cls"] == "mild")
    return {"total": len(ALERTS), "severe": severe, "mild": mild}


@app.websocket("/ws/alerts")
async def ws_alerts(ws: WebSocket):
    await ws.accept()
    await ws.send_json({"type": "hello", "msg": "real-time alert stream"})
    while True:
        await ws.receive_text()
        await ws.send_json({"latest": ALERTS[-10:]})
