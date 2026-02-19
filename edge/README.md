# Edge Layer

- Simulates STM32L476/nRF52840 constraints.
- 125Hz acquisition, 10s window, 5s overlap.
- Bandpass+z-score expected from training pipeline.
- Alert rules:
  - Mild: confidence >0.70 for 3 consecutive windows.
  - Severe: confidence >0.85 single window.
- AES-128 packet encryption for edge→fog BLE-like payload.
