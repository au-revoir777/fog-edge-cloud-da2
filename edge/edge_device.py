"""Phase 4: TinyML edge device simulation with BLE-like encrypted alert packets."""
from __future__ import annotations

import base64
import json
import os
import time
from dataclasses import dataclass
from typing import Deque, Dict, List
from collections import deque

import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

FS = 125
WINDOW = 1250
STEP = 625


@dataclass
class EdgeAlert:
    device_id: str
    timestamp: float
    confidence: float
    cls: str
    reliability_score: float
    raw_window: List[float] | None = None


class TinyMLEdgeDevice:
    def __init__(self, device_id: str, key: bytes):
        self.device_id = device_id
        self.key = key
        self.buffer: Deque[float] = deque(maxlen=WINDOW)
        self.mild_counter = 0
        self.reliability_score = 0.95

    def ingest(self, sample: float) -> None:
        self.buffer.append(sample)

    def infer(self) -> Dict[str, float]:
        arr = np.array(self.buffer)
        energy = float(np.mean(np.abs(arr)))
        hypoxia_score = min(max(0.4 + energy, 0.0), 0.99)
        if hypoxia_score > 0.85:
            cls = "severe"
        elif hypoxia_score > 0.70:
            cls = "mild"
        else:
            cls = "normal"
        return {"confidence": hypoxia_score, "class": cls, "latency_ms": float(np.random.uniform(27, 45))}

    def step_window(self) -> EdgeAlert | None:
        if len(self.buffer) < WINDOW:
            return None
        pred = self.infer()
        if pred["class"] == "severe":
            self.mild_counter = 0
            return EdgeAlert(self.device_id, time.time(), pred["confidence"], "severe", self.reliability_score, list(self.buffer))
        if pred["class"] == "mild":
            self.mild_counter += 1
            if self.mild_counter >= 3:
                self.mild_counter = 0
                return EdgeAlert(self.device_id, time.time(), pred["confidence"], "mild", self.reliability_score)
        else:
            self.mild_counter = 0
        return None

    def encrypt_packet(self, alert: EdgeAlert) -> bytes:
        packet = {
            "device_id": alert.device_id,
            "timestamp": alert.timestamp,
            "confidence": alert.confidence,
            "class": alert.cls,
            "reliability_score": alert.reliability_score,
        }
        if alert.raw_window is not None:
            packet["raw_window"] = alert.raw_window
        nonce = os.urandom(12)
        cipher = AESGCM(self.key)
        ct = cipher.encrypt(nonce, json.dumps(packet).encode(), None)
        return base64.b64encode(nonce + ct)


def simulate_device(device_id: str = "edge-001", seconds: int = 60) -> List[bytes]:
    dev = TinyMLEdgeDevice(device_id, AESGCM.generate_key(bit_length=128))
    packets = []
    for i in range(seconds * FS):
        value = np.sin(2 * np.pi * 1.2 * i / FS) + np.random.normal(0, 0.2)
        if i % (20 * FS) < 3 * FS:
            value += 0.7
        dev.ingest(value)
        if i % STEP == 0 and i > WINDOW:
            alert = dev.step_window()
            if alert:
                packets.append(dev.encrypt_packet(alert))
    return packets


if __name__ == "__main__":
    alerts = simulate_device()
    print(f"generated_encrypted_alerts={len(alerts)}")
