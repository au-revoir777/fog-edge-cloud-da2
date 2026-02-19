# Smart Remote Respiratory Monitoring Using TinyML-Enabled Edge-Fog-Cloud Architecture

Production-grade prototype implementing hierarchical healthcare IoT monitoring.

## Folder Structure
- `edge/` TinyML edge simulation (alert-only transmission + AES-128 packets)
- `fog/` Aggregation, false-positive reduction, queue/caching
- `cloud/` FastAPI APIs, WebSocket, InfluxDB/PostgreSQL-ready deployment
- `training/` Dataset preprocessing + 1D CNN training and quantization
- `simulation/` Baseline comparison, confusion matrix, ROC, energy/CPU metrics
- `dashboard/` Real-time dashboard UI scaffold
- `docs/` Architecture, deployment, security, training/performance reports

## Run end-to-end phases
```bash
python run_all_phases.py
```

## Key constraints tracked
- Edge inference <= 50 ms
- End-to-end latency <= 200 ms
- Bandwidth reduction >= 80%
- INT8 TinyML envelope <= 100 KB SRAM / <= 500 KB Flash
- Accuracy >= 92%, severe recall >=95%, severe precision >=90%

See generated artifacts:
- `training/artifacts/phase1_metrics.json`
- `training/artifacts/phase2_3_metrics.json`
- `simulation/artifacts/benchmark_results.json`
- `simulation/artifacts/final_validation.json`
