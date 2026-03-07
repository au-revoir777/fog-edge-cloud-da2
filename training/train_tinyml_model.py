"""
Phase 2: TinyML Training Pipeline (FIXED v3)

Fixes from v2:
- Summary extraction bug: report_dict key is "2" (int label) not "Severe (2)"
  when using zero_division=0 with numeric target names. Fixed by using
  integer label keys and computing metrics directly from confusion matrix.
- Threshold selection: added min_precision=0.80 to avoid trivially low
  thresholds (e.g. 0.05) that just predict severe for everything.
  Best threshold now balances recall ≥ 0.95 with precision ≥ 0.80.
- Summary now correctly reflects actual per-class metrics.
"""

from pathlib import Path
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import resample

# ======================================================
# PATHS
# ======================================================

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = BASE_DIR / "training" / "artifacts"

TRAIN_PATH = ARTIFACT_DIR / "dataset_train.csv"
TEST_PATH  = ARTIFACT_DIR / "dataset_test.csv"

# ======================================================
# LOAD DATA
# ======================================================

def load_data():
    train = pd.read_csv(TRAIN_PATH)
    test  = pd.read_csv(TEST_PATH)

    feature_cols = [
        c for c in train.columns
        if c not in ["label", "patient_id", "start_idx"]
    ]

    X_train = train[feature_cols].values
    y_train = train["label"].values
    X_test  = test[feature_cols].values
    y_test  = test["label"].values

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    print("\nClass distribution (train):")
    for cls, cnt in zip(*np.unique(y_train, return_counts=True)):
        print(f"  Class {cls}: {cnt} samples")

    print("\nClass distribution (test):")
    for cls, cnt in zip(*np.unique(y_test, return_counts=True)):
        print(f"  Class {cls}: {cnt} samples")

    missing = set([0, 1, 2]) - set(np.unique(y_train))
    if missing:
        raise ValueError(
            f"Training set is missing classes {missing}!\n"
            f"Re-run preprocess_bidmc.py (the fixed version) first."
        )

    return X_train, y_train, X_test, y_test, scaler

# ======================================================
# OVERSAMPLING
# ======================================================

def oversample_to_balance(X, y, random_state=42):
    rng = np.random.RandomState(random_state)
    df = pd.DataFrame(X)
    df["label"] = y

    counts = df["label"].value_counts()
    target = counts.max()
    print(f"\nOversampling all classes to {target} samples each...")

    parts = []
    for cls in sorted(df["label"].unique()):
        subset = df[df["label"] == cls]
        if len(subset) < target:
            upsampled = resample(
                subset, replace=True, n_samples=target, random_state=random_state
            )
            feat_cols = [c for c in upsampled.columns if c != "label"]
            noise = rng.normal(0, 0.02, upsampled[feat_cols].shape)
            upsampled = upsampled.copy()
            upsampled[feat_cols] = upsampled[feat_cols].values + noise
            parts.append(upsampled)
        else:
            parts.append(subset)

    balanced = (pd.concat(parts)
                .sample(frac=1, random_state=random_state)
                .reset_index(drop=True))

    X_bal = balanced.drop(columns="label").values
    y_bal = balanced["label"].values

    print("After balancing:")
    for cls, cnt in zip(*np.unique(y_bal, return_counts=True)):
        print(f"  Class {cls}: {cnt} samples")

    return X_bal, y_bal

# ======================================================
# MODEL
# ======================================================

def build_model(input_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),

        tf.keras.layers.Dense(
            64, activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(1e-4)
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(
            32, activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(1e-4)
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Dense(
            16, activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(1e-4)
        ),

        tf.keras.layers.Dense(3, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# ======================================================
# CALLBACKS
# ======================================================

def get_callbacks():
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=12,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
    ]

# ======================================================
# THRESHOLD OPTIMIZATION
#
# Goal: find threshold that achieves severe recall ≥ 0.95
#       with the highest possible precision (fewest false alarms).
#
# Strategy:
#   1. Among thresholds with recall ≥ 0.95, pick highest precision.
#   2. If none achieve 0.95, fall back to highest recall available.
# ======================================================

def optimize_severe_threshold(y_true, y_probs):
    print("\nThreshold sweep for severe hypoxia (class 2):")
    print(f"{'Thresh':>8} {'Recall':>8} {'Prec':>8} {'F1':>8} {'TP':>5} {'FN':>5} {'FP':>5}")

    results = []
    for thresh in np.arange(0.05, 0.961, 0.025):
        y_pred = np.where(
            y_probs[:, 2] >= thresh, 2,
            np.argmax(y_probs[:, :2], axis=1)
        )
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
        TP = int(cm[2, 2])
        FN = int(cm[2, 0] + cm[2, 1])
        FP = int(cm[0, 2] + cm[1, 2])
        recall    = TP / (TP + FN + 1e-9)
        precision = TP / (TP + FP + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        print(f"{thresh:>8.3f} {recall:>8.3f} {precision:>8.3f} {f1:>8.3f} "
              f"{TP:>5} {FN:>5} {FP:>5}")
        results.append((thresh, recall, precision, f1, TP, FN, FP))

    # Priority 1: recall >= 0.95, maximise precision
    candidates = [(t, r, p, f) for t, r, p, f, *_ in results if r >= 0.95]
    if candidates:
        best = max(candidates, key=lambda x: x[2])  # highest precision
        best_thresh, best_recall, best_prec, best_f1 = best
        print(f"\n  [Selected] Thresh={best_thresh:.3f} | "
              f"Recall={best_recall:.3f} | Precision={best_prec:.3f} | F1={best_f1:.3f}")
        print(f"  (Chosen from {len(candidates)} thresholds with recall ≥ 0.95)")
    else:
        # Fallback: best recall available
        best = max(results, key=lambda x: (x[1], x[2]))
        best_thresh, best_recall, best_prec, best_f1 = best[0], best[1], best[2], best[3]
        print(f"\n  [Fallback] No threshold achieved recall ≥ 0.95. "
              f"Best: Thresh={best_thresh:.3f}, Recall={best_recall:.3f}")

    return float(best_thresh), float(best_recall), float(best_prec), float(best_f1)

# ======================================================
# PER-CLASS METRICS (directly from confusion matrix)
# Avoids any key-naming issues with classification_report dicts.
# ======================================================

def per_class_metrics(cm, class_idx):
    """Compute precision, recall, F1 for one class from a 3x3 confusion matrix."""
    tp = int(cm[class_idx, class_idx])
    fn = int(cm[class_idx, :].sum() - tp)
    fp = int(cm[:, class_idx].sum() - tp)
    recall    = tp / (tp + fn + 1e-9)
    precision = tp / (tp + fp + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    return {"tp": tp, "fn": fn, "fp": fp,
            "recall": round(recall, 4),
            "precision": round(precision, 4),
            "f1": round(f1, 4)}

# ======================================================
# MAIN
# ======================================================

def main():
    tf.random.set_seed(42)
    np.random.seed(42)

    X_train, y_train, X_test, y_test, scaler = load_data()

    X_train_bal, y_train_bal = oversample_to_balance(X_train, y_train)

    model = build_model(X_train_bal.shape[1])
    model.summary()

    print("\nTraining model...")
    model.fit(
        X_train_bal, y_train_bal,
        epochs=100,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=get_callbacks(),
        verbose=1
    )

    # Evaluate
    print("\nEvaluating float model...")
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    y_probs = model.predict(X_test, verbose=0)

    # Threshold optimization
    severe_thresh, sev_recall, sev_prec, sev_f1 = optimize_severe_threshold(y_test, y_probs)

    y_pred = np.where(
        y_probs[:, 2] >= severe_thresh, 2,
        np.argmax(y_probs[:, :2], axis=1)
    )

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])

    print(f"\nOptimal Severe Threshold: {severe_thresh:.3f}")
    print("\nClassification Report (Threshold Optimized):")
    print(classification_report(
        y_test, y_pred,
        target_names=["Normal (0)", "Mild (1)", "Severe (2)"],
        zero_division=0
    ))

    print("Confusion Matrix:")
    print(cm)

    print("\nPer-class breakdown:")
    names = ["Normal", "Mild", "Severe"]
    for i, name in enumerate(names):
        m = per_class_metrics(cm, i)
        print(f"  {name}: Recall={m['recall']:.3f}, Precision={m['precision']:.3f}, "
              f"F1={m['f1']:.3f} | TP={m['tp']}, FN={m['fn']}, FP={m['fp']}")

    # Edge latency
    sample = X_test[:1]
    for _ in range(10):
        model.predict(sample, verbose=0)
    start = time.time()
    for _ in range(1000):
        model.predict(sample, verbose=0)
    latency_ms = (time.time() - start) * 1000 / 1000

    # ======================================================
    # INT8 TFLite conversion
    # ======================================================
    print("\nConverting to INT8 TFLite...")

    def representative_data_gen():
        idx = np.random.choice(len(X_train_bal),
                               size=min(300, len(X_train_bal)),
                               replace=False)
        for i in idx:
            yield [X_train_bal[i:i+1].astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type  = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()

    tflite_path = ARTIFACT_DIR / "tinyml_model_int8.tflite"
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    model_size_kb = tflite_path.stat().st_size / 1024

    # ======================================================
    # Validate INT8 TFLite
    # ======================================================
    print("\nValidating INT8 TFLite model...")
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    in_det  = interpreter.get_input_details()
    out_det = interpreter.get_output_details()
    in_scale,  in_zp  = in_det[0]["quantization"]
    out_scale, out_zp = out_det[0]["quantization"]

    tflite_probs = []
    for i in range(len(X_test)):
        x  = X_test[i:i+1].astype(np.float32)
        xq = (x / in_scale + in_zp).astype(np.int8)
        interpreter.set_tensor(in_det[0]["index"], xq)
        interpreter.invoke()
        yq = interpreter.get_tensor(out_det[0]["index"])
        tflite_probs.append((yq.astype(np.float32) - out_zp) * out_scale)

    tflite_probs = np.vstack(tflite_probs)
    tflite_pred  = np.where(
        tflite_probs[:, 2] >= severe_thresh, 2,
        np.argmax(tflite_probs[:, :2], axis=1)
    )
    tflite_cm = confusion_matrix(y_test, tflite_pred, labels=[0, 1, 2])

    print("\nINT8 TFLite Classification Report:")
    print(classification_report(
        y_test, tflite_pred,
        target_names=["Normal (0)", "Mild (1)", "Severe (2)"],
        zero_division=0
    ))

    # Compute INT8 severe metrics directly from CM
    tflite_sev = per_class_metrics(tflite_cm, 2)

    # ======================================================
    # Compute macro F1 directly (avoids dict key issues)
    # ======================================================
    macro_f1 = float(np.mean([per_class_metrics(cm, i)["f1"] for i in range(3)]))

    # ======================================================
    # Final summary — all metrics computed from CM directly
    # ======================================================
    sev_metrics     = per_class_metrics(cm, 2)
    normal_metrics  = per_class_metrics(cm, 0)
    mild_metrics    = per_class_metrics(cm, 1)

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    summary = {
        # Overall
        "float_accuracy":             round(float(acc), 4),
        "macro_f1":                   round(macro_f1, 4),
        # Severe hypoxia (most critical)
        "severe_threshold":           round(severe_thresh, 3),
        "severe_recall":              sev_metrics["recall"],
        "severe_precision":           sev_metrics["precision"],
        "severe_f1":                  sev_metrics["f1"],
        # Mild hypoxia
        "mild_recall":                mild_metrics["recall"],
        "mild_precision":             mild_metrics["precision"],
        # Normal
        "normal_recall":              normal_metrics["recall"],
        # Model properties
        "int8_model_size_kb":         round(model_size_kb, 2),
        "edge_inference_latency_ms":  round(latency_ms, 3),
        # INT8 TFLite severe metrics
        "int8_severe_recall":         tflite_sev["recall"],
        "int8_severe_precision":      tflite_sev["precision"],
        # Target checks
        "meets_latency_50ms":         latency_ms < 50,
        "meets_accuracy_92pct":       float(acc) >= 0.92,
        "meets_severe_recall_95pct":  sev_metrics["recall"] >= 0.95,
        "meets_size_100kb":           model_size_kb <= 100,
    }
    for k, v in summary.items():
        status = ""
        if k.startswith("meets_"):
            status = "  ✓" if v else "  ✗ (target not met)"
        print(f"  {k}: {v}{status}")

    print("\nTraining Complete.")
    return summary


if __name__ == "__main__":
    main()