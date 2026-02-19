# Cloud Layer

- Kubernetes-ready services for storage, APIs, analytics, retraining.
- Datastores:
  - InfluxDB 2.7 for 1-second granularity time-series.
  - PostgreSQL 15 for metadata.
- Interfaces:
  - REST: `/api/alerts`, `/api/models`, `/api/devices`, `/api/analytics`.
  - WebSocket: `/ws/alerts` for real-time feed.
- MQTT over TLS 1.3 expected for fog->cloud uplink.
