# Model Training Report

- Dataset source target: PhysioNet BIDMC PPG and Respiration Dataset (125Hz).
- Windowing: 10s windows with 5s overlap.
- Labels: Normal (>90%), Mild (<=90%), Severe (<=85%) based on SpO2.
- TinyML model: mandated 1D CNN architecture.
- Optimization: 40% prune target + INT8 quantization.
- Reported metrics are stored in `training/artifacts/phase2_3_metrics.json`.
