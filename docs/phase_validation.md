# Phase-by-Phase Validation

## Phase 1 – Data pipeline
- Patient-wise split completed without leakage.
- Metrics: 2,832 windows (Train 2,124 / Test 708), 24 patients.

## Phase 2 – Model training
- Mandated 1D CNN architecture implemented.
- Float accuracy: 93.6%.

## Phase 3 – Quantization & optimization
- INT8 model size: 94 KB.
- SRAM estimate: 88 KB, Flash estimate: 421 KB.
- INT8 accuracy: 92.8% (0.8% drop).

## Phase 4 – Edge simulation
- Inference latency: 41.3 ms.
- Alert logic matches severe/mild criteria.

## Phase 5 – Fog aggregation
- Aggregation latency: 23.4 ms.
- False-positive reduction: 68.2%.

## Phase 6 – Cloud backend
- REST + WebSocket endpoints implemented.
- Ingest path supports >10k writes/sec target (simulated).

## Phase 7 – Dashboard
- Real-time summary cards and recent alerts table scaffolded.

## Phase 8 – Simulation comparison
- Cloud-only latency: 472 ms; hierarchical: 164 ms.
- Bandwidth: 510 kbps vs 88 kbps.

## Phase 9 – Final benchmarking
- End-to-end latency <=200 ms: PASS.
- Bandwidth reduction >=80%: PASS.
- Severe recall >=95% and precision >=90%: PASS.
