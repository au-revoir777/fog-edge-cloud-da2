# Security Design

- **Edge→Fog**: AES-128 (GCM) encrypted BLE-like payloads, minimal PHI transfer.
- **Fog→Cloud**: MQTT over TLS 1.3 with X.509 device identity.
- **At-rest encryption**: DB volume encryption required for PHI.
- **Privacy principle**: continuous raw signals stay local; only alerts/metadata sent, plus severe event raw window for clinician verification.
- **Reliability**: local caching at fog and edge offline buffering to keep loss <=0.1%.
