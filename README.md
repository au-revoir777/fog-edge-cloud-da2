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
│       ├── phase1_metrics.json     # DA-1 simulation metrics
│       ├── phase2_3_metrics.json   # DA-2 intermediate evaluation metrics
│       ├── tinyml_metrics.json     # Final model metrics summary
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
├── DA2_Report_Final.docx           # Full DA-2 submission report
└── README.md
```

---

## Dataset

**Source:** PhysioNet BIDMC PPG and Respiration Dataset  
**Link:** https://physionet.org/content/bidmc/1.0.0/bidmc_csv/  
**Reference:** Goldberger et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet. *Circulation*, 101(23), e215–e220.

53 ICU patients | 125 Hz | 8-minute windows per patient  
Signals used: SpO₂, PPG amplitude, Respiratory Impedance

Download the CSV files from PhysioNet and place them in a local directory before running `data_pipeline.py`. The preprocessed `dataset_train.csv` and `dataset_test.csv` are already included in this repository.

**Class labels:**

| Label | Class | SpO₂ Range | Train Samples | Test Samples |
|-------|-------|------------|---------------|--------------|
| 0 | Normal | ≥ 95% | 3,697 | 1,140 |
| 1 | Mild Hypoxia | 90–94% | 66 | 21 |
| 2 | Severe Hypoxia | < 90% | 78 | 25 |

---

## Setup

```bash
pip install tensorflow scikit-learn imbalanced-learn pandas numpy matplotlib
```

Tested on Python 3.11, TensorFlow 2.x, Windows 10/11.

---

## Running the Pipeline

### Step 1 — Preprocess raw BIDMC signals (only needed if regenerating datasets)
```bash
cd training
python data_pipeline.py
```
Produces `artifacts/dataset_train.csv` and `artifacts/dataset_test.csv`.

### Step 2 — Train and evaluate the TinyML model
```bash
cd training
python -m train_tinyml_model
```

This will:
1. Run stratified 5-fold cross-validation on the training set
2. Train the final two-stage model on the full training set
3. Evaluate on the held-out test set
4. Convert both stage models to INT8 TFLite
5. Save all 7 evaluation plots to `artifacts/plots/`

Training takes approximately 5–10 minutes on CPU (no GPU required).

---

## Model Architecture

### Two-Stage MLP Classifier

**Why two stages?** A single 3-class softmax cannot specialise its decision boundaries when one class has only 66 training samples. Separating the problem into two binary decisions allows each stage to focus on its own boundary independently.

```
Input (29 features)
        │
        ▼
┌─────────────────────────────────┐
│  Stage 1: Normal vs Abnormal    │  threshold = 0.30
│  Dense(32)→BN→Dropout(0.3)     │  → Normal  : stop, no alert
│  Dense(16)→BN→Dropout(0.2)     │  → Abnormal: pass to Stage 2
│  Dense(8)→Dense(2, softmax)    │
└─────────────────────────────────┘
        │ (abnormal only)
        ▼
┌─────────────────────────────────┐
│  Stage 2: Mild vs Severe        │  threshold = 0.475 (val-selected)
│  Dense(32)→BN→Dropout(0.3)     │  → Mild   : moderate alert
│  Dense(16)→BN→Dropout(0.2)     │  → Severe : immediate escalation
│  Dense(8)→Dense(2, softmax)    │
└─────────────────────────────────┘
```

**Loss function:** Focal Loss (γ = 2.0) — down-weights easy Normal samples, focusing gradient on the hard hypoxic boundary  
**Imbalance handling:** SMOTE oversampling on training folds only + class weights  
**Validation strategy:** Stratified 5-Fold CV; thresholds selected on validation set; test set held out completely

---

## Results Summary

| Metric | Achieved | Target | Status |
|--------|----------|--------|--------|
| Overall Accuracy | 98.4% | ≥ 92% | ✓ Pass |
| Severe Hypoxia Recall | 100% (CV: 100% ± 0%) | ≥ 95% | ✓ Pass |
| Severe Hypoxia Precision | ~69–71% | ≥ 90% | See note |
| Total Model Size (INT8) | 12.70 KB | ≤ 100 KB | ✓ Pass |
| Inference Latency (STM32 est.) | ~35–45 ms | ≤ 50 ms | ✓ Pass |

**Note on severe precision:** The 90% precision target was set in DA-1 before the dataset's class distribution was fully characterised. With only 66 genuine mild and 78 genuine severe training samples, the mild/severe decision boundary is inherently ambiguous. All mild misclassifications are escalated to severe (never to normal), making this a clinically safe error direction. The fog layer's temporal correlation — requiring 3 consecutive severe detections to confirm an alert — mitigates false positives in deployment.

---

## Evaluation Plots

| Figure | Description |
|--------|-------------|
| fig1_class_distribution | Raw class imbalance in train and test sets |
| fig2_stage1_training_curves | Stage 1 accuracy and focal loss over epochs |
| fig3_stage2_training_curves | Stage 2 training — shows dataset ceiling at epoch 7 |
| fig4_confusion_matrices | Float32 vs INT8 confusion matrices side-by-side |
| fig5_cv_fold_performance | Per-fold metrics across all 5 CV folds with NFR targets |
| fig6_threshold_sweep | Recall/precision/F1 vs threshold for Stage 2 |
| fig7_nfr_compliance | NFR compliance summary bar chart |

---

## Target Hardware

**Device:** STM32L476 (Cortex-M4F, 80 MHz, 128 KB SRAM, 1 MB Flash)  
**Runtime:** TensorFlow Lite Micro  

| Component | Required | Budget |
|-----------|----------|--------|
| Stage 1 + Stage 2 models | 12.70 KB flash | 1 MB flash |
| TFLite Micro runtime | ~50 KB SRAM | 128 KB SRAM |
| Signal buffer (10s × 3ch) | ~15 KB SRAM | — |
| RTOS + stack | ~20 KB SRAM | — |
| **Total** | **~91 KB SRAM** | **128 KB SRAM** |

---

## References

- Goldberger et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet. *Circulation*, 101(23).
- David et al. (2021). TensorFlow Lite Micro. *MLSys 2021*.
- Lin et al. (2017). Focal Loss for Dense Object Detection. *IEEE ICCV*.
- Chawla et al. (2002). SMOTE: Synthetic Minority Over-sampling Technique. *JAIR*, 16.
- Warden & Situnayake (2019). *TinyML*. O'Reilly Media.
