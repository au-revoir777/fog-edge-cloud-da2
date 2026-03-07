# Smart Remote Respiratory Monitoring Using TinyML-Enabled Edge-Fog-Cloud Architecture

**Course:** BCSE313L — Fundamentals of Fog and Edge Computing  
**Semester:** Winter 2025-26 | Slot: C1/TC1  
**Authors:** A Sahana (23BCE5122), Roshan Parveen (23BCE1256)  
**Submitted to:** Dr. Priyadarshini R

---

## Project Overview

This project implements a TinyML-based respiratory monitoring system for wearable edge devices. A two-stage MLP classifier detects hypoxic events in real time from SpO₂, PPG, and respiratory impedance signals, operating within the memory and latency constraints of an STM32L476 microcontroller.

The system is designed as the edge inference layer of a three-tier Edge-Fog-Cloud architecture, where edge devices perform local classification, fog nodes aggregate and filter alerts, and the cloud handles long-term analytics and model retraining.

---

## Repository Structure

```
fog-edge/
├── training/
│   ├── train_tinyml_model.py       # Main training pipeline (v6)
│   ├── data_pipeline.py            # Preprocessing: filtering, windowing,
│   │                               #   feature extraction, train/test split
│   ├── model_training_res.txt      # Full terminal output from final training run
│   └── artifacts/
│       ├── dataset_train.csv       # Training set (3,841 samples, 29 features)
│       ├── dataset_test.csv        # Held-out test set (1,186 samples, 29 features)
│       │                           #   Natural class distribution — never used
│       │                           #   during training or threshold selection
│       ├── phase1_metrics.json     
│       ├── phase2_3_metrics.json   
│       ├── stage1_normal_vs_abnormal_int8.tflite
│       │                           # Stage 1 INT8 model: Normal vs Abnormal
│       │                           #   Size: 6.38 KB | Input: 29 float32 features
│       ├── stage2_mild_vs_severe_int8.tflite
│       │                           # Stage 2 INT8 model: Mild vs Severe Hypoxia
│       │                           #   Size: 6.33 KB | Only runs on abnormal samples
│       └── plots/
│           ├── fig1_class_distribution.png
│           ├── fig2_stage1_training_curves.png
│           ├── fig3_stage2_training_curves.png
│           ├── fig4_confusion_matrices.png
│           ├── fig5_cv_fold_performance.png
│           ├── fig6_threshold_sweep.png
│           └── fig7_nfr_compliance.png
├── 23BCE5122_23BCE1256_DA2.pdf           # Full DA-2 submission report
└── README.md
```

## References

- Goldberger et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet. *Circulation*, 101(23).
- David et al. (2021). TensorFlow Lite Micro. *MLSys 2021*.
- Lin et al. (2017). Focal Loss for Dense Object Detection. *IEEE ICCV*.
- Chawla et al. (2002). SMOTE: Synthetic Minority Over-sampling Technique. *JAIR*, 16.
- Warden & Situnayake (2019). *TinyML*. O'Reilly Media.
