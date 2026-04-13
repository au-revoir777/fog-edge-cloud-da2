"""
step1_pre_score.py
==================
DA3 — Pre-score with real INT8 TFLite models → predictions.json

Architecture (clean separation):
  Edge layer  — EdgeDevice_01/02/03: each runs full two-stage TFLite inference.
                Publishes ONLY edge-level fields: spo2, hr, rr, edge_label,
                edge_confidence, p_abnormal, p_severe, severity_code,
                top_trigger_feature, trigger_feature_zscore.

  Fog layer   — FogGateway_01: aggregates all three devices.
                Publishes ONLY fog-level fields: adaptive_threshold,
                spo2_drop_rate, fog_decision_reason, fog_tier,
                fog_alert_fired, active_devices_in_window, ward_anomaly,
                ward_anomaly_reason.

  NO fog fields appear in edge device payloads.
  NO edge inference fields appear in the gateway payload.

Novel additions:
  1. AdaptiveFogFilter  — drop-rate-aware confirmation threshold (fog layer)
  2. CrossPatientMonitor — correlated multi-device ward anomaly (fog layer)
  3. Feature importance  — top z-score feature name per alert (edge layer)

Run once:
    python3 step1_pre_score.py

Outputs:
    predictions.json  — one record per frame:
        "edge_devices"  : { EdgeDevice_01: {...}, EdgeDevice_02: {...},
                            EdgeDevice_03: {...} }   ← edge-only fields
        "fog_gateway"   : { FogGateway_01: {...} }   ← fog-only fields
"""

import json, time, warnings, collections
import numpy as np
import tensorflow as tf

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── load models ────────────────────────────────────────────────────────────────
print("Loading TFLite models …")
s1 = tf.lite.Interpreter("stage1_normal_vs_abnormal_int8.tflite")
s2 = tf.lite.Interpreter("stage2_mild_vs_severe_int8.tflite")
s1.allocate_tensors(); s2.allocate_tensors()
s1_in  = s1.get_input_details()[0];  s1_out = s1.get_output_details()[0]
s2_in  = s2.get_input_details()[0];  s2_out = s2.get_output_details()[0]
print(f"  Stage 1  scale={s1_in['quantization'][0]:.5f}  zp={s1_in['quantization'][1]}")
print(f"  Stage 2  scale={s2_in['quantization'][0]:.5f}  zp={s2_in['quantization'][1]}")

def _dq(x, s, z): return (x.astype(np.float32) - z) * s
def _sm(x): e = np.exp(x - x.max()); return e / e.sum()
def _q(f, s, z): return np.clip(np.round(f / s + z), -128, 127).astype(np.int8)

def infer(features_f32):
    """Two-stage INT8 inference. Returns (label, confidence, p_abnormal, p_severe)."""
    sc1, zp1 = s1_in['quantization']
    s1.set_tensor(s1_in['index'], _q(features_f32, sc1, zp1).reshape(1, 29))
    s1.invoke()
    p1 = _sm(_dq(s1.get_tensor(s1_out['index'])[0], *s1_out['quantization']))
    p_abn = float(p1[1])
    if p_abn < 0.30:
        return "Normal", float(p1[0]), p_abn, 0.0
    sc2, zp2 = s2_in['quantization']
    s2.set_tensor(s2_in['index'], _q(features_f32, sc2, zp2).reshape(1, 29))
    s2.invoke()
    p2 = _sm(_dq(s2.get_tensor(s2_out['index'])[0], *s2_out['quantization']))
    p_sev = float(p2[1])
    if p_sev > 0.50:
        return "Severe", p_sev, p_abn, p_sev
    return "Mild", float(p2[0]), p_abn, p_sev

# ── load verified feature pools ────────────────────────────────────────────────
print("Loading verified feature pools …")
with open("feature_pools.json") as fp:
    raw_pools = json.load(fp)

pools = {k: [np.array(v["f"], dtype=np.float32) for v in vals]
         for k, vals in raw_pools.items()}
print(f"  Normal: {len(pools['Normal'])}  "
      f"Mild: {len(pools['Mild'])}  "
      f"Severe: {len(pools['Severe'])}")

pool_idx = {k: 0 for k in pools}

FEATURE_NAMES = [
    "spo2_mean", "spo2_variance", "spo2_std", "spo2_min", "spo2_max",
    "spo2_range", "spo2_zcr", "spo2_rms", "spo2_skewness", "spo2_kurtosis",
    "ppg_mean", "ppg_variance", "ppg_std", "ppg_min", "ppg_max",
    "ppg_range", "ppg_zcr", "ppg_rms", "ppg_skewness", "ppg_kurtosis",
    "resp_spectral_power", "resp_dominant_freq", "resp_spectral_entropy",
    "resp_psd_ratio", "resp_rms", "resp_zcr", "resp_variance",
    "spo2_ppg_correlation", "estimated_rr",
]

_normal_stack = np.stack([v for v in pools["Normal"]])
_normal_mean  = _normal_stack.mean(axis=0)
_normal_std   = _normal_stack.std(axis=0) + 1e-6

# Separate pool indices per device so each device draws independently
pool_idx_per_device = {
    "EdgeDevice_01": {k: 0 for k in pools},
    "EdgeDevice_02": {k: 0 for k in pools},
    "EdgeDevice_03": {k: 0 for k in pools},
}

def next_feature(label, device_id):
    p   = pools[label]
    idx = pool_idx_per_device[device_id]
    f   = p[idx[label] % len(p)]
    idx[label] += 1
    return f + np.random.randn(29).astype(np.float32) * 0.05

def top_trigger_feature(features_f32, label):
    """Novel Addition 3: Feature importance via z-score (edge layer only)."""
    if label == "Normal":
        return "n/a", 0.0
    z   = np.abs((features_f32 - _normal_mean) / _normal_std)
    idx = int(np.argmax(z))
    return FEATURE_NAMES[idx], round(float(z[idx]), 2)

def display_vitals(label):
    """Map model label to realistic SpO2/HR/RR display values."""
    if label == "Normal":
        return (round(np.random.uniform(95.5, 99.5), 1),
                int(np.random.uniform(60, 88)),
                round(np.random.uniform(12, 18), 1))
    if label == "Mild":
        return (round(np.random.uniform(90.0, 94.9), 1),
                int(np.random.uniform(80, 108)),
                round(np.random.uniform(18, 25), 1))
    return (round(np.random.uniform(80.0, 89.9), 1),
            int(np.random.uniform(100, 130)),
            round(np.random.uniform(25, 36), 1))

# ── build one full edge payload (EDGE-ONLY fields) ─────────────────────────────
def make_edge_payload(device_id, true_label, window_index):
    """
    Runs two-stage TFLite inference for one device.
    Returns a dict containing ONLY edge-layer fields.
    No fog fields (ward_anomaly, adaptive_threshold, etc.) appear here.
    """
    f = next_feature(true_label, device_id)
    label, conf, p_abn, p_sev = infer(f)
    spo2, hr, rr = display_vitals(label)
    trigger_feat, trigger_zscore = top_trigger_feature(f, label)

    # severity_code at the edge is the raw model output, NOT fog-confirmed.
    # The fog layer decides whether to escalate/suppress mild alerts.
    raw_severity = 2 if label == "Severe" else (1 if label == "Mild" else 0)

    return {
        # vitals
        "spo2":                   spo2,
        "heart_rate":             hr,
        "respiratory_rate":       rr,
        # model outputs — edge layer
        "edge_label":             label,
        "edge_confidence":        round(conf, 3),
        "p_abnormal":             round(p_abn, 3),
        "p_severe":               round(p_sev, 3),
        "raw_severity_code":      raw_severity,  # pre-fog, unconfirmed
        # Novel Addition 3: feature attribution (edge layer)
        "top_trigger_feature":    trigger_feat,
        "trigger_feature_zscore": round(trigger_zscore, 2),
        # metadata
        "device_id":              device_id,
        "fog_node_id":            "FogGateway_01",
        "model_size_kb":          12.7,
        "window_index":           window_index,
    }, spo2, raw_severity

# ══════════════════════════════════════════════════════════════════════════════
# NOVEL ADDITION 1 — Adaptive Fog Filter (fog layer)
# Operates on EdgeDevice_01 (primary patient) SpO2 stream.
# The output (adaptive_threshold, fog_tier, fog_alert_fired, severity_code)
# lives ONLY in FogGateway_01's payload.
# ══════════════════════════════════════════════════════════════════════════════
class AdaptiveFogFilter:
    RAPID_DROP    = 2.0
    MODERATE_DROP = 1.0

    def __init__(self):
        self.consec_mild = 0
        self.prev_spo2   = None

    def _compute_threshold(self, spo2):
        if self.prev_spo2 is None:
            return 3, 0.0
        drop = max(self.prev_spo2 - spo2, 0.0)
        if drop >= self.RAPID_DROP:
            return 1, round(drop, 2)
        elif drop >= self.MODERATE_DROP:
            return 2, round(drop, 2)
        return 3, round(drop, 2)

    def process(self, label, conf, spo2):
        threshold, drop_rate = self._compute_threshold(spo2)
        self.prev_spo2 = spo2

        if label == "Normal":
            self.consec_mild = 0
            return (label, conf, "edge_local", False,
                    threshold, drop_rate, "normal_reading_reset")
        if label == "Severe":
            self.consec_mild = 0
            return (label, conf, "fog_escalated_immediate", True,
                    threshold, drop_rate, "severe_bypass_immediate")

        self.consec_mild += 1
        if threshold == 1:
            reason = f"rapid_drop_{drop_rate}pct_threshold_lowered_to_1"
        elif threshold == 2:
            reason = f"moderate_drop_{drop_rate}pct_threshold_lowered_to_2"
        else:
            reason = "stable_standard_threshold_3"
        confirmed = self.consec_mild >= threshold
        tier = "fog_confirmed_adaptive" if confirmed else "fog_pending"
        return (label, conf, tier, confirmed,
                threshold, drop_rate, reason)


# ══════════════════════════════════════════════════════════════════════════════
# NOVEL ADDITION 2 — Cross-Patient Monitor (fog layer)
# Uses all three device severity_codes to detect ward-level anomalies.
# Output lives ONLY in FogGateway_01's payload.
# ══════════════════════════════════════════════════════════════════════════════
class CrossPatientMonitor:
    WINDOW_FRAMES     = 12   # 60s at 5s/frame
    ANOMALY_THRESHOLD = 3    # all 3 devices abnormal

    def __init__(self):
        self._event_window = collections.deque()

    def update(self, frame_idx, device_severity_map):
        """
        device_severity_map: { device_id: raw_severity_code }
        Returns (active_devices_in_window, ward_anomaly, ward_anomaly_reason)
        """
        cutoff = frame_idx - self.WINDOW_FRAMES
        while self._event_window and self._event_window[0][0] <= cutoff:
            self._event_window.popleft()

        for dev_id, sc in device_severity_map.items():
            if sc >= 1:
                self._event_window.append((frame_idx, dev_id))

        active_devices = len(set(ev[1] for ev in self._event_window))
        ward_anomaly   = int(active_devices >= self.ANOMALY_THRESHOLD)
        reason = (f"{active_devices}_devices_abnormal_in_60s_window_"
                  f"possible_environmental_factor"
                  if ward_anomaly else "normal_cross_patient_state")
        return active_devices, ward_anomaly, reason


# ── clinical session timeline ──────────────────────────────────────────────────
# EdgeDevice_01 drives the primary scenario; 02 and 03 follow correlated patterns.
sequence = (
    ["Normal"] * 30 +
    ["Mild"]   * 5  +
    ["Normal"] * 15 +
    ["Mild"]   * 8  +
    ["Normal"] * 12 +
    ["Severe"] * 4  +
    ["Normal"] * 8  +
    ["Mild"]   * 4  +
    ["Normal"] * 10 +
    ["Severe"] * 3  +
    ["Normal"] * 12 +
    ["Mild"]   * 6  +
    ["Severe"] * 3  +
    ["Mild"]   * 3  +
    ["Normal"] * 14 +
    ["Severe"] * 2  +
    ["Normal"] * 10
)

def companion_true_label(primary_label):
    """
    Generate a correlated but independent true label for a companion device.
    Companion devices have their own patients — they correlate but don't mirror.
    """
    if primary_label == "Normal":
        return "Normal"
    if primary_label == "Mild":
        return np.random.choice(["Mild", "Normal"], p=[0.40, 0.60])
    # Severe primary
    return np.random.choice(["Severe", "Mild", "Normal"], p=[0.35, 0.30, 0.35])


fog     = AdaptiveFogFilter()
monitor = CrossPatientMonitor()
records = []
base_ts = int(time.time()) - len(sequence) * 5

print(f"\nGenerating {len(sequence)}-frame telemetry session …")
print(f"  All 3 edge devices run full TFLite inference independently.")
print(f"  Fog metrics computed at FogGateway_01 and kept separate.\n")

for i, true_lbl_01 in enumerate(sequence):
    ts = (base_ts + i * 5) * 1000

    # ── EdgeDevice_01: full TFLite inference ───────────────────────────────
    dev01, spo2_01, sc_01 = make_edge_payload("EdgeDevice_01", true_lbl_01, i)

    # ── EdgeDevice_02: independent TFLite inference (correlated label) ─────
    true_lbl_02 = companion_true_label(true_lbl_01)
    dev02, _,      sc_02 = make_edge_payload("EdgeDevice_02", true_lbl_02, i)

    # ── EdgeDevice_03: independent TFLite inference (correlated label) ─────
    true_lbl_03 = companion_true_label(true_lbl_01)
    dev03, _,      sc_03 = make_edge_payload("EdgeDevice_03", true_lbl_03, i)

    # ── FogGateway_01: fog logic applied to aggregated readings ────────────
    # Novel Addition 1: Adaptive fog filter tracks EdgeDevice_01 (primary)
    (fog_lbl, fog_conf, fog_tier, fog_alert,
     adapt_thresh, drop_rate, fog_reason) = fog.process(
        dev01["edge_label"], dev01["edge_confidence"], spo2_01)

    # Fog-confirmed severity for EdgeDevice_01 (mild needs confirmation)
    fog_severity_01 = (2 if fog_lbl == "Severe" else
                       1 if (fog_lbl == "Mild" and fog_alert) else 0)

    # Novel Addition 2: Cross-patient monitor uses raw edge severity of all 3
    active_devs, ward_anomaly, ward_reason = monitor.update(i, {
        "EdgeDevice_01": sc_01,
        "EdgeDevice_02": sc_02,
        "EdgeDevice_03": sc_03,
    })

    # ── FogGateway_01 payload: FOG-ONLY fields ─────────────────────────────
    fog_payload = {
        # Novel Addition 1 — Adaptive Fog Filter
        "adaptive_threshold":       adapt_thresh,
        "spo2_drop_rate":           drop_rate,
        "fog_decision_reason":      fog_reason,
        "fog_tier":                 fog_tier,
        "fog_alert_fired":          int(fog_alert),
        "fog_confirmed_severity":   fog_severity_01,  # fog decision on Dev01
        # Novel Addition 2 — Cross-Patient Ward Monitor
        "active_devices_in_window": active_devs,
        "ward_anomaly":             ward_anomaly,
        "ward_anomaly_reason":      ward_reason,
        # metadata
        "fog_node_id":              "FogGateway_01",
        "primary_device":           "EdgeDevice_01",
    }

    records.append({
        "ts": ts,
        "edge_devices": {
            "EdgeDevice_01": dev01,
            "EdgeDevice_02": dev02,
            "EdgeDevice_03": dev03,
        },
        "fog_gateway": {
            "FogGateway_01": fog_payload,
        },
    })

with open("predictions.json", "w") as fp:
    json.dump(records, fp, indent=2)

# ── summary ────────────────────────────────────────────────────────────────────
fog_alerts  = sum(r["fog_gateway"]["FogGateway_01"]["fog_alert_fired"] for r in records)
ward_alerts = sum(r["fog_gateway"]["FogGateway_01"]["ward_anomaly"]    for r in records)

def label_counts(device_key):
    counts = {}
    for r in records:
        lbl = r["edge_devices"][device_key]["edge_label"]
        counts[lbl] = counts.get(lbl, 0) + 1
    return counts

print(f"\n{'─'*62}")
print(f"  Frames total  : {len(records)}")
for dev in ["EdgeDevice_01", "EdgeDevice_02", "EdgeDevice_03"]:
    c = label_counts(dev)
    print(f"  {dev}: Normal={c.get('Normal',0)}  "
          f"Mild={c.get('Mild',0)}  Severe={c.get('Severe',0)}")

thresh_counts = {1: 0, 2: 0, 3: 0}
for r in records:
    t = r["fog_gateway"]["FogGateway_01"]["adaptive_threshold"]
    thresh_counts[t] = thresh_counts.get(t, 0) + 1

print(f"{'─'*62}")
print(f"  [Novel 1 — Fog] Adaptive threshold breakdown:")
for t, cnt in sorted(thresh_counts.items()):
    lbl_t = {1: "rapid drop ", 2: "moderate  ", 3: "standard  "}[t]
    print(f"    threshold={t} ({lbl_t}): {cnt:3d} frames")
print(f"  [Novel 1 — Fog] Fog alerts fired     : {fog_alerts}")
print(f"  [Novel 2 — Fog] Ward anomaly frames  : {ward_alerts}")

top_feats = {}
for r in records:
    feat = r["edge_devices"]["EdgeDevice_01"]["top_trigger_feature"]
    if feat != "n/a":
        top_feats[feat] = top_feats.get(feat, 0) + 1
if top_feats:
    top3 = sorted(top_feats.items(), key=lambda x: -x[1])[:3]
    print(f"  [Novel 3 — Edge] Top trigger features (Dev01): "
          + "  |  ".join(f"{f}({c})" for f, c in top3))
print(f"{'─'*62}")
print(f"\n✓  predictions.json — clean separation:")
print(f"   edge_devices.EdgeDevice_01/02/03  ← edge-only TFLite fields")
print(f"   fog_gateway.FogGateway_01         ← fog-only aggregation fields")
print(f"\n   Next → python3 step2_replay_to_thingsboard.py")