# Fog Layer

- Raspberry Pi 4 simulation.
- Ingests encrypted alerts from 10-25 edge devices per gateway.
- 3-window mild temporal smoothing.
- Severe alerts bypass smoothing (priority queue).
- Reliability-weighted score:
  `sum(confidence_i * reliability_i) / sum(reliability_i)`.
- Redis local cache + queue depth up to 10,000 during cloud disconnect.
