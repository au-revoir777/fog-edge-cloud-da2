"""
step2_replay_to_thingsboard.py
================================
DA3 — Fog Gateway replay via ThingsBoard Gateway API

Clean separation of edge vs fog telemetry:

  EdgeDevice_01/02/03  publish ONLY edge-layer fields:
    spo2, heart_rate, respiratory_rate,
    edge_label, edge_confidence, p_abnormal, p_severe,
    raw_severity_code, top_trigger_feature, trigger_feature_zscore

  FogGateway_01 publishes ONLY fog-layer fields:
    adaptive_threshold, spo2_drop_rate, fog_decision_reason,
    fog_tier, fog_alert_fired, fog_confirmed_severity,
    active_devices_in_window, ward_anomaly, ward_anomaly_reason

This means:
  - ThingsBoard's EdgeDevice dashboards show only edge metrics.
  - ThingsBoard's FogGateway dashboard shows only fog/aggregation metrics.
  - Rule chain routing (edge vs fog alarms) maps cleanly to device type.

Prerequisites:
    1. FogGateway_01 created in ThingsBoard with "Is gateway" = ON
    2. Paste FogGateway_01 access token into FOG_TOKEN below
    3. pip install paho-mqtt

Usage:
    python3 step2_replay_to_thingsboard.py
    python3 step2_replay_to_thingsboard.py --speed 3
    python3 step2_replay_to_thingsboard.py --speed 0   # instant bulk load
"""

import json, time, sys, argparse
import paho.mqtt.client as mqtt

# ╔══════════════════════════════════════════════════════════════╗
# ║  CONFIGURE THESE BEFORE RUNNING                              ║
# ╠══════════════════════════════════════════════════════════════╣
THINGSBOARD_HOST = "thingsboard.cloud"   # or "localhost" for Docker
THINGSBOARD_PORT = 1883
FOG_TOKEN        = "EIQf3Q5hyKcITVgquuqX"   # FogGateway_01 token
# ╚══════════════════════════════════════════════════════════════╝

GATEWAY_TOPIC    = "v1/gateway/telemetry"
PREDICTIONS_FILE = "predictions.json"

COLOUR = {
    "Normal": "\033[92m", "Mild": "\033[93m", "Severe": "\033[91m",
    "RESET":  "\033[0m",
}

def col(label, text):
    return f"{COLOUR.get(label, '')}{text}{COLOUR['RESET']}"

def on_connect(client, userdata, flags, rc):
    codes = {0: "Connected", 1: "Bad protocol", 2: "Client ID rejected",
             3: "Server unavailable", 4: "Bad credentials", 5: "Not authorised"}
    if rc == 0:
        print(f"  ✓  Fog gateway connected → {THINGSBOARD_HOST}:{THINGSBOARD_PORT}")
    else:
        print(f"  ✗  Connect failed: {codes.get(rc, rc)}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Replay speed (1=real-time 5s, 3=faster, 0=instant)")
    args = parser.parse_args()

    print("\n━━━  DA3 Fog Gateway Replay — Clean Edge/Fog Separation  ━━━")
    print(f"  Gateway : FogGateway_01")
    print(f"  Devices : EdgeDevice_01/02/03  (all run TFLite, edge fields only)")
    print(f"  Gateway : FogGateway_01        (fog aggregation fields only)")
    print(f"  Topic   : {GATEWAY_TOPIC}")
    print(f"  Speed   : {args.speed}x")
    print(f"  NOTE    : Live server timestamps — data appears in ThingsBoard")
    print(f"            'Last 5 min' / 'Last hour' windows\n")

    with open(PREDICTIONS_FILE) as f:
        records = json.load(f)
    print(f"  Loaded {len(records)} frames from {PREDICTIONS_FILE}")

    # Validate new format
    first = records[0]
    if "edge_devices" not in first or "fog_gateway" not in first:
        print("\n  ✗  ERROR: predictions.json uses the old format.")
        print("     Re-run step1_pre_score.py first, then retry.\n")
        sys.exit(1)

    required_edge_keys = {
        "spo2", "heart_rate", "respiratory_rate",
        "edge_label", "edge_confidence", "p_abnormal", "p_severe",
        "raw_severity_code", "top_trigger_feature", "trigger_feature_zscore",
    }
    required_fog_keys = {
        "adaptive_threshold", "spo2_drop_rate", "fog_decision_reason",
        "fog_tier", "fog_alert_fired", "fog_confirmed_severity",
        "active_devices_in_window", "ward_anomaly", "ward_anomaly_reason",
    }

    # Verify separation — no fog keys in edge payloads, no edge keys in fog payload
    fog_keys_in_edge = set()
    edge_keys_in_fog = set()
    for r in records[:5]:
        for dev in ["EdgeDevice_01", "EdgeDevice_02", "EdgeDevice_03"]:
            fog_keys_in_edge |= required_fog_keys & set(r["edge_devices"][dev].keys())
        edge_keys_in_fog |= required_edge_keys & set(r["fog_gateway"]["FogGateway_01"].keys())

    if fog_keys_in_edge:
        print(f"\n  ✗  ERROR: Fog fields found in edge payloads: {fog_keys_in_edge}")
        print("     Re-run step1_pre_score.py to regenerate.\n")
        sys.exit(1)
    if edge_keys_in_fog:
        print(f"\n  ✗  ERROR: Edge fields found in fog payload: {edge_keys_in_fog}")
        print("     Re-run step1_pre_score.py to regenerate.\n")
        sys.exit(1)

    print("  ✓  Edge/fog field separation verified")

    # Distribution summary
    for dev in ["EdgeDevice_01", "EdgeDevice_02", "EdgeDevice_03"]:
        totals = {}
        for r in records:
            lbl = r["edge_devices"][dev]["edge_label"]
            totals[lbl] = totals.get(lbl, 0) + 1
        summary = "  ".join(f"{lbl}={cnt}" for lbl, cnt in sorted(totals.items()))
        print(f"    {dev}: {summary}")

    # MQTT connect
    client = mqtt.Client()
    client.username_pw_set(FOG_TOKEN)
    client.on_connect = on_connect
    client.connect(THINGSBOARD_HOST, THINGSBOARD_PORT, keepalive=60)
    client.loop_start()
    time.sleep(1)

    delay = 5.0 / args.speed if args.speed > 0 else 0

    hdr = (f"{'#':>4}  {'Dev01 SpO2':>10}  {'Label':12}  "
           f"{'Dev02 SpO2':>10}  {'Dev03 SpO2':>10}  "
           f"{'Ward':>5}  {'Fog Tier'}")
    print(f"\n  {hdr}")
    print("  " + "─" * 100)

    i = 0
    try:
        for i, record in enumerate(records):
            edges = record["edge_devices"]
            fog   = record["fog_gateway"]["FogGateway_01"]

            # Live timestamp so ThingsBoard time windows work correctly
            live_ts = int(time.time() * 1000)

            # ── ThingsBoard Gateway API payload ───────────────────────────
            # Each device entry contains ONLY that device's own fields.
            # FogGateway_01 entry contains ONLY fog aggregation fields.
            gateway_payload = {
                "EdgeDevice_01": [{"ts": live_ts, "values": edges["EdgeDevice_01"]}],
                "EdgeDevice_02": [{"ts": live_ts, "values": edges["EdgeDevice_02"]}],
                "EdgeDevice_03": [{"ts": live_ts, "values": edges["EdgeDevice_03"]}],
                "FogGateway_01": [{"ts": live_ts, "values": fog}],
            }

            payload = json.dumps(gateway_payload)
            result  = client.publish(GATEWAY_TOPIC, payload, qos=1)
            result.wait_for_publish()

            # ── console row ────────────────────────────────────────────────
            d1 = edges["EdgeDevice_01"]
            d2 = edges["EdgeDevice_02"]
            d3 = edges["EdgeDevice_03"]
            ward = "X" if fog["ward_anomaly"] else " "

            print("  {:>4}  {:18s}  {:12s}  {:18s}  {:18s}  {}  {}".format(
                i + 1,
                col(d1["edge_label"], "{:>5.1f}%".format(d1["spo2"])),
                col(d1["edge_label"], "{:12s}".format(d1["edge_label"])),
                col(d2["edge_label"], "{:>5.1f}%".format(d2["spo2"])),
                col(d3["edge_label"], "{:>5.1f}%".format(d3["spo2"])),
                ward, fog["fog_tier"],
            ))

            if delay > 0:
                time.sleep(delay)

    except KeyboardInterrupt:
        print("\n  Stopped by user.")
    finally:
        client.loop_stop()
        client.disconnect()

        alerts     = sum(r["fog_gateway"]["FogGateway_01"]["fog_alert_fired"]
                         for r in records[:i + 1])
        ward_total = sum(r["fog_gateway"]["FogGateway_01"]["ward_anomaly"]
                         for r in records[:i + 1])

        print(f"\n  Sent {i + 1}/{len(records)} frames via FogGateway_01")
        print(f"  Fog alerts (fog_alert_fired)  : {alerts}")
        print(f"  Ward anomaly frames           : {ward_total}")
        print(f"\n  ThingsBoard devices auto-created (each shows its own fields):")
        print(f"    EdgeDevice_01  — edge TFLite fields (spo2, edge_label, confidence …)")
        print(f"    EdgeDevice_02  — edge TFLite fields (spo2, edge_label, confidence …)")
        print(f"    EdgeDevice_03  — edge TFLite fields (spo2, edge_label, confidence …)")
        print(f"    FogGateway_01  — fog aggregation fields (ward_anomaly, adaptive …)")
        print(f"\n  Replay complete.")
        print(f"  Dashboard time window: 'Last 5 minutes' or 'Last hour'\n")

if __name__ == "__main__":
    main()