"""Phase 5: Fog aggregation, false-positive reduction, priority queue, and local cache."""
from __future__ import annotations

import base64
import json
import queue
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

try:
    import redis
    REDIS_AVAILABLE = True
except Exception:
    REDIS_AVAILABLE = False


@dataclass
class Alert:
    device_id: str
    timestamp: float
    confidence: float
    cls: str
    reliability_score: float


class FogAggregator:
    def __init__(self, key: bytes):
        self.key = key
        self.history = defaultdict(lambda: deque(maxlen=3))
        self.reliability = defaultdict(lambda: 0.95)
        self.priority_queue: "queue.PriorityQueue[tuple[int, Alert]]" = queue.PriorityQueue(maxsize=10000)
        self.local_cache: List[dict] = []
        if REDIS_AVAILABLE:
            try:
                self.redis = redis.Redis(host="localhost", port=6379, decode_responses=True)
                self.redis.ping()
            except Exception:
                self.redis = None
        else:
            self.redis = None

    def decrypt_packet(self, payload: bytes) -> Alert:
        raw = base64.b64decode(payload)
        nonce, ct = raw[:12], raw[12:]
        data = AESGCM(self.key).decrypt(nonce, ct, None)
        packet = json.loads(data.decode())
        return Alert(
            device_id=packet["device_id"],
            timestamp=packet["timestamp"],
            confidence=packet["confidence"],
            cls=packet["class"],
            reliability_score=packet["reliability_score"],
        )

    def aggregated_score(self, alerts: List[Alert]) -> float:
        top = sum(a.confidence * a.reliability_score for a in alerts)
        bot = sum(a.reliability_score for a in alerts) + 1e-9
        return top / bot

    def process_alert(self, alert: Alert) -> bool:
        if alert.cls == "severe":
            self.priority_queue.put((0, alert))
            self.cache(alert, confirmed=True)
            return True
        h = self.history[alert.device_id]
        h.append(alert)
        if len(h) == 3 and all(a.cls == "mild" and a.confidence > 0.70 for a in h):
            score = self.aggregated_score(list(h))
            confirmed = score > 0.72
            if confirmed:
                self.priority_queue.put((1, alert))
            self.cache(alert, confirmed=confirmed)
            return confirmed
        return False

    def cache(self, alert: Alert, confirmed: bool) -> None:
        item = {"device": alert.device_id, "ts": alert.timestamp, "class": alert.cls, "conf": alert.confidence, "confirmed": confirmed}
        self.local_cache.append(item)
        if self.redis is not None:
            self.redis.rpush("fog_alert_cache", json.dumps(item))

    def flush_to_cloud_payload(self) -> List[dict]:
        out = []
        while not self.priority_queue.empty():
            _, alert = self.priority_queue.get()
            out.append(alert.__dict__)
        return out


def simulate_fog_roundtrip(encrypted_packets: List[bytes], key: bytes) -> Dict[str, float]:
    fog = FogAggregator(key)
    start = time.perf_counter()
    total = 0
    confirmed = 0
    for p in encrypted_packets:
        total += 1
        a = fog.decrypt_packet(p)
        if fog.process_alert(a):
            confirmed += 1
    ms = (time.perf_counter() - start) * 1000
    return {
        "fog_aggregation_latency_ms": round(ms / max(total, 1), 2),
        "alerts_received": total,
        "alerts_confirmed": confirmed,
        "false_positive_reduction_pct": round(100 * (1 - confirmed / max(total, 1)), 2),
    }
