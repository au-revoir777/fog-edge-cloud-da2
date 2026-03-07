# Training Pipeline

## Phase 1
Run `python training/data_pipeline.py` to generate patient-wise split window dataset (10s windows, 5s overlap).

## Phase 2/3
Run `python training/train_tinyml_model.py` to train mandated 1D CNN and export metrics/INT8 artifact.

Model architecture:
- Conv1D(32,k=5,s=2)+BN+ReLU
- Conv1D(64,k=3,s=2)+BN+ReLU
- Conv1D(128,k=3,s=2)+BN+ReLU
- GAP
- Dense(64)+Dropout(0.3)+ReLU
- Dense(3,softmax)
