"""
Phase 2: TinyML Training Pipeline (v6)

Changes over v5:
- Full visualisation suite saved to ARTIFACT_DIR/plots/
  Figure 1 — Class distribution (train vs test)
  Figure 2 — Stage 1 training curves (accuracy + loss)
  Figure 3 — Stage 2 training curves (accuracy + loss)
  Figure 4 — Confusion matrix: Float32 vs INT8 side-by-side
  Figure 5 — Cross-validation fold performance (bar + error bands)
  Figure 6 — Stage 2 threshold sweep (recall / precision / F1)
  Figure 7 — NFR compliance summary (horizontal bar chart)
- History objects are now captured from train_fold() and passed
  to the plotting functions.
- All other logic identical to v5.
"""

from pathlib import Path
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    warnings.warn(
        "imbalanced-learn not found. Falling back to random oversampling.\n"
        "Install with: pip install imbalanced-learn"
    )
    HAS_SMOTE = False

warnings.filterwarnings("ignore", category=FutureWarning)

# ======================================================
# PATHS & CONFIG
# ======================================================

BASE_DIR     = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = BASE_DIR / "training" / "artifacts"
PLOT_DIR     = ARTIFACT_DIR / "plots"
TRAIN_PATH   = ARTIFACT_DIR / "dataset_train.csv"
TEST_PATH    = ARTIFACT_DIR / "dataset_test.csv"

RANDOM_SEED   = 42
N_CV_FOLDS    = 5
VAL_FRACTION  = 0.15
FOCAL_GAMMA   = 2.0
STAGE1_THRESH = 0.30

# Colour palette used across all plots
C_NORMAL  = "#4C9BE8"   # blue
C_MILD    = "#F5A623"   # amber
C_SEVERE  = "#E84C4C"   # red
C_TRAIN   = "#5B8DD9"
C_VAL     = "#F07B3F"
C_PASS    = "#4CAF50"
C_FAIL    = "#E84C4C"
C_NEUTRAL = "#90A4AE"

# ======================================================
# PLOT HELPERS
# ======================================================

def _save(fig, name):
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot: {path}")


def _style_ax(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linestyle="--")


# ======================================================
# FIGURE 1 — CLASS DISTRIBUTION
# ======================================================

def plot_class_distribution(y_train, y_test):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    fig.suptitle("Dataset Class Distribution", fontsize=13, fontweight="bold", y=1.01)

    labels     = ["Normal (0)", "Mild Hypoxia (1)", "Severe Hypoxia (2)"]
    colours    = [C_NORMAL, C_MILD, C_SEVERE]
    label_idx  = [0, 1, 2]

    for ax, y, title in zip(axes, [y_train, y_test], ["Training Set", "Test Set (held-out)"]):
        counts = [np.sum(y == i) for i in label_idx]
        bars   = ax.bar(labels, counts, color=colours, edgecolor="white", linewidth=1.2, width=0.55)
        for bar, cnt in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(counts) * 0.02,
                    str(cnt), ha="center", va="bottom", fontsize=9, fontweight="bold")
        _style_ax(ax, title, "Class", "Sample Count")
        ax.set_ylim(0, max(counts) * 1.18)
        ax.set_xticklabels(labels, rotation=12, ha="right")

    fig.tight_layout()
    _save(fig, "fig1_class_distribution.png")


# ======================================================
# FIGURE 2 & 3 — TRAINING CURVES
# ======================================================

def plot_training_curves(history, stage_name, fig_name):
    """Accuracy and loss curves for one stage."""
    hist  = history.history
    epochs = range(1, len(hist["loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle(f"{stage_name} — Training Curves", fontsize=13, fontweight="bold", y=1.01)

    # --- Accuracy ---
    ax1.plot(epochs, hist["accuracy"],     color=C_TRAIN, lw=2,   label="Train accuracy")
    ax1.plot(epochs, hist["val_accuracy"], color=C_VAL,   lw=2,   label="Val accuracy",   linestyle="--")
    best_epoch = int(np.argmin(hist["val_loss"])) + 1
    ax1.axvline(best_epoch, color="grey", linestyle=":", lw=1.2, label=f"Best epoch ({best_epoch})")
    ax1.set_ylim(max(0, min(hist["accuracy"] + hist["val_accuracy"]) - 0.05), 1.02)
    _style_ax(ax1, "Accuracy", "Epoch", "Accuracy")
    ax1.legend(fontsize=8)

    # --- Loss ---
    ax2.plot(epochs, hist["loss"],     color=C_TRAIN, lw=2,   label="Train loss")
    ax2.plot(epochs, hist["val_loss"], color=C_VAL,   lw=2,   label="Val loss",   linestyle="--")
    ax2.axvline(best_epoch, color="grey", linestyle=":", lw=1.2, label=f"Best epoch ({best_epoch})")

    # Mark LR reduction events (val_loss plateau points)
    lr_key = "learning_rate" if "learning_rate" in hist else None
    if lr_key:
        lrs = hist[lr_key]
        reductions = [i + 1 for i in range(1, len(lrs)) if lrs[i] < lrs[i - 1]]
        for ep in reductions:
            ax2.axvline(ep, color=C_MILD, linestyle=":", lw=1.0, alpha=0.7)
        if reductions:
            ax2.plot([], [], color=C_MILD, linestyle=":", lw=1.0,
                     label=f"LR reduced ({len(reductions)}×)")

    _style_ax(ax2, "Focal Loss", "Epoch", "Loss")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    _save(fig, fig_name)


# ======================================================
# FIGURE 4 — CONFUSION MATRICES (Float32 vs INT8)
# ======================================================

def plot_confusion_matrices(cm_float, cm_int8, y_test):
    class_names = ["Normal", "Mild\nHypoxia", "Severe\nHypoxia"]
    fig, axes   = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle("Confusion Matrices — Float32 vs INT8 TFLite (Test Set)",
                 fontsize=13, fontweight="bold", y=1.01)

    for ax, cm, title in zip(axes, [cm_float, cm_int8],
                              ["Float32 Model", "INT8 TFLite Model"]):
        n      = cm.sum()
        im     = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(class_names, fontsize=9)
        ax.set_yticklabels(class_names, fontsize=9)
        ax.set_xlabel("Predicted", fontsize=9)
        ax.set_ylabel("True", fontsize=9)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)

        thresh = cm.max() / 2
        for i in range(3):
            for j in range(3):
                pct = cm[i, j] / (cm[i].sum() + 1e-9) * 100
                ax.text(j, i, f"{cm[i,j]}\n({pct:.0f}%)",
                        ha="center", va="center", fontsize=9,
                        color="white" if cm[i, j] > thresh else "black",
                        fontweight="bold" if i == j else "normal")

        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)

    fig.tight_layout()
    _save(fig, "fig4_confusion_matrices.png")


# ======================================================
# FIGURE 5 — CROSS-VALIDATION FOLD PERFORMANCE
# ======================================================

def plot_cv_results(cv_df):
    metrics = {
        "accuracy":      ("Overall Accuracy",    C_NEUTRAL),
        "sev_recall":    ("Severe Recall",        C_SEVERE),
        "sev_precision": ("Severe Precision",     C_MILD),
        "mild_recall":   ("Mild Recall",          C_NORMAL),
    }
    folds  = np.arange(1, len(cv_df) + 1)
    n      = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(14, 4), sharey=False)
    fig.suptitle(f"Stratified {N_CV_FOLDS}-Fold Cross-Validation Results",
                 fontsize=13, fontweight="bold", y=1.01)

    nfr_lines = {
        "accuracy":      0.92,
        "sev_recall":    0.95,
        "sev_precision": 0.90,
        "mild_recall":   0.95,
    }

    for ax, (col, (label, colour)) in zip(axes, metrics.items()):
        vals = cv_df[col].values
        mean = vals.mean()
        std  = vals.std()

        bars = ax.bar(folds, vals, color=colour, alpha=0.75,
                      edgecolor="white", linewidth=1.1, width=0.6)
        ax.axhline(mean, color=colour, lw=2, linestyle="-",  label=f"Mean {mean:.3f}")
        ax.axhline(mean + std, color=colour, lw=1, linestyle="--", alpha=0.5)
        ax.axhline(mean - std, color=colour, lw=1, linestyle="--", alpha=0.5,
                   label=f"±1 SD ({std:.3f})")

        target = nfr_lines.get(col)
        if target:
            ax.axhline(target, color="red", lw=1.2, linestyle=":",
                       label=f"NFR target ({target})")

        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    v + 0.005, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=7.5)

        _style_ax(ax, label, "Fold", "Score")
        ax.set_xticks(folds)
        ax.set_ylim(max(0, min(vals) - 0.15), 1.08)
        ax.legend(fontsize=7, loc="lower right")

    fig.tight_layout()
    _save(fig, "fig5_cv_fold_performance.png")


# ======================================================
# FIGURE 6 — STAGE 2 THRESHOLD SWEEP
# ======================================================

def plot_threshold_sweep(y_true, s2_model, X_abn_test,
                         abn_mask, stage2_thresh):
    """
    Re-runs the threshold sweep on the test set purely for
    visualisation. The selected threshold was chosen on val set.
    """
    s2_probs = s2_model.predict(X_abn_test, verbose=0)
    thresholds = np.arange(0.05, 0.961, 0.025)
    recalls, precisions, f1s = [], [], []

    for thresh in thresholds:
        abn_pred = np.where(s2_probs[:, 1] >= thresh, 2, 1)
        y_pred   = np.zeros(len(y_true), dtype=np.int32)
        y_pred[abn_mask] = abn_pred
        cm  = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
        m   = per_class_metrics(cm, 2)
        recalls.append(m["recall"])
        precisions.append(m["precision"])
        f1s.append(m["f1"])

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(thresholds, recalls,    color=C_SEVERE, lw=2,   label="Severe Recall")
    ax.plot(thresholds, precisions, color=C_MILD,   lw=2,   label="Severe Precision")
    ax.plot(thresholds, f1s,        color=C_NORMAL, lw=2,   label="Severe F1",   linestyle="--")

    ax.axvline(stage2_thresh, color="grey",  linestyle=":", lw=1.5,
               label=f"Selected threshold ({stage2_thresh:.3f})")
    ax.axhline(0.95, color=C_SEVERE, linestyle=":", lw=1.0, alpha=0.6,
               label="Recall NFR target (0.95)")
    ax.axhline(0.90, color=C_MILD,   linestyle=":", lw=1.0, alpha=0.6,
               label="Precision NFR target (0.90)")

    ax.fill_between(thresholds, recalls, 0, color=C_SEVERE, alpha=0.05)
    _style_ax(ax, "Stage 2 Threshold Sweep — Severe Hypoxia Detection\n(visualised on test set)",
              "Threshold", "Score")
    ax.set_xlim(0.05, 0.95)
    ax.set_ylim(0, 1.08)
    ax.legend(fontsize=9)

    fig.tight_layout()
    _save(fig, "fig6_threshold_sweep.png")


# ======================================================
# FIGURE 7 — NFR COMPLIANCE SUMMARY
# ======================================================

def plot_nfr_compliance(results):
    nfrs = [
        ("Overall Accuracy",    results["accuracy"],        0.92,  True,  "%"),
        ("Severe Recall",       results["sev_recall"],      0.95,  True,  "%"),
        ("Severe Precision",    results["sev_precision"],   0.90,  True,  "%"),
        ("Model Size (total)",  results["total_kb"],        100.0, False, "KB"),
        ("Inference Latency*",  results["latency_ms"],      50.0,  False, "ms"),
    ]

    labels, values, targets, higher_is_better, units = zip(*nfrs)
    n = len(nfrs)

    # Normalise each metric to 0-1 against its target for bar length
    norm_vals    = []
    norm_targets = []
    colours      = []
    for val, tgt, hib in zip(values, targets, higher_is_better):
        if hib:
            nv = val / tgt        # >1 means passing
            nt = 1.0
            colours.append(C_PASS if val >= tgt else C_FAIL)
        else:
            nv = tgt / val        # >1 means passing (smaller is better)
            nt = 1.0
            colours.append(C_PASS if val <= tgt else C_FAIL)
        norm_vals.append(min(nv, 2.0))   # cap at 2× for display
        norm_targets.append(nt)

    fig, ax = plt.subplots(figsize=(10, 5))
    y_pos = np.arange(n)

    bars = ax.barh(y_pos, norm_vals, color=colours, alpha=0.80,
                   edgecolor="white", linewidth=1.1, height=0.55)
    ax.axvline(1.0, color="black", lw=1.5, linestyle="--", label="NFR target (1.0×)")

    # Value labels
    for i, (bar, val, tgt, unit) in enumerate(zip(bars, values, targets, units)):
        display = f"{val*100:.1f}{unit}" if unit == "%" else f"{val:.1f} {unit}"
        tgt_str = f"{tgt*100:.0f}%" if unit == "%" else f"{tgt:.0f} {unit}"
        status  = "✓" if colours[i] == C_PASS else "✗"
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{status}  {display}  (target: {tgt_str})",
                va="center", fontsize=9,
                color=C_PASS if colours[i] == C_PASS else C_FAIL,
                fontweight="bold")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlim(0, 2.4)
    ax.set_xlabel("Achievement vs Target (1.0 = exactly at target, >1.0 = passing)", fontsize=9)
    ax.set_title("NFR Compliance Summary\n* Latency measured in Python/CPU — actual STM32 latency will be lower",
                 fontsize=11, fontweight="bold", pad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=8)

    pass_patch = mpatches.Patch(color=C_PASS, label="Passing")
    fail_patch = mpatches.Patch(color=C_FAIL, label="Not met")
    ax.legend(handles=[pass_patch, fail_patch], fontsize=9, loc="lower right")

    fig.tight_layout()
    _save(fig, "fig7_nfr_compliance.png")


# ======================================================
# FOCAL LOSS
# ======================================================

def focal_loss(gamma=FOCAL_GAMMA):
    def loss_fn(y_true, y_pred):
        y_true   = tf.cast(y_true, tf.int32)
        y_true_oh = tf.one_hot(y_true, depth=y_pred.shape[-1])
        y_pred   = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        ce       = -tf.reduce_sum(y_true_oh * tf.math.log(y_pred), axis=-1)
        p_t      = tf.reduce_sum(y_true_oh * y_pred, axis=-1)
        return tf.reduce_mean(tf.pow(1.0 - p_t, gamma) * ce)
    loss_fn.__name__ = f"focal_loss_g{gamma}"
    return loss_fn


# ======================================================
# LOAD DATA
# ======================================================

def load_data():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df  = pd.read_csv(TEST_PATH)
    skip     = {"label", "patient_id", "start_idx"}
    feat     = [c for c in train_df.columns if c not in skip]

    X_train  = train_df[feat].values
    y_train  = train_df["label"].values
    X_test   = test_df[feat].values
    y_test   = test_df["label"].values

    print("\nRaw class distribution (train):")
    for cls, cnt in zip(*np.unique(y_train, return_counts=True)):
        print(f"  Class {cls}: {cnt} samples")
    print("\nClass distribution (test — held out):")
    for cls, cnt in zip(*np.unique(y_test, return_counts=True)):
        print(f"  Class {cls}: {cnt} samples")

    missing = set([0, 1, 2]) - set(np.unique(y_train))
    if missing:
        raise ValueError(f"Training set missing classes {missing}.")

    return X_train, y_train, X_test, y_test, feat


# ======================================================
# OVERSAMPLING
# ======================================================

def _smote(X, y):
    min_count = int(np.bincount(y).min())
    k = min(5, min_count - 1)
    if k < 1:
        return _random_oversample(X, y)
    return SMOTE(random_state=RANDOM_SEED, k_neighbors=k).fit_resample(X, y)


def _random_oversample(X, y, noise_std=0.02):
    rng    = np.random.RandomState(RANDOM_SEED)
    target = np.bincount(y).max()
    parts  = []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        if len(idx) < target:
            extra   = rng.choice(idx, size=target - len(idx), replace=True)
            X_extra = X[extra] + rng.normal(0, noise_std, (len(extra), X.shape[1]))
            parts.append((np.vstack([X[idx], X_extra]),
                          np.full(target, cls, dtype=y.dtype)))
        else:
            parts.append((X[idx], y[idx]))
    Xb  = np.vstack([p[0] for p in parts])
    yb  = np.concatenate([p[1] for p in parts])
    idx = rng.permutation(len(yb))
    return Xb[idx], yb[idx]


def oversample(X, y):
    fn     = _smote if HAS_SMOTE else _random_oversample
    Xb, yb = fn(X, y)
    method = "SMOTE" if HAS_SMOTE else "random+noise"
    print(f"  Oversampled ({method}):")
    for cls, cnt in zip(*np.unique(yb, return_counts=True)):
        print(f"    Class {cls}: {cnt} samples")
    return Xb, yb


# ======================================================
# MODELS
# ======================================================

def build_stage1(input_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(32, activation="relu",
                              kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(16, activation="relu",
                              kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(8,  activation="relu"),
        tf.keras.layers.Dense(2,  activation="softmax"),
    ], name="stage1_normal_vs_abnormal")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss=focal_loss(FOCAL_GAMMA), metrics=["accuracy"])
    return model


def build_stage2(input_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(32, activation="relu",
                              kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(16, activation="relu",
                              kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(8,  activation="relu"),
        tf.keras.layers.Dense(2,  activation="softmax"),
    ], name="stage2_mild_vs_severe")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss=focal_loss(FOCAL_GAMMA), metrics=["accuracy"])
    return model


def get_callbacks(monitor="val_loss"):
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor, patience=12,
            restore_best_weights=True, verbose=0),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor, factor=0.5,
            patience=5, min_lr=1e-6, verbose=0),
    ]


# ======================================================
# DATA HELPERS
# ======================================================

def make_stage1_labels(y):
    return (y > 0).astype(np.int32)


def make_stage2_data(X, y):
    mask = y > 0
    return X[mask], (y[mask] - 1).astype(np.int32)


# ======================================================
# PER-CLASS METRICS
# ======================================================

def per_class_metrics(cm, class_idx):
    tp   = int(cm[class_idx, class_idx])
    fn   = int(cm[class_idx, :].sum() - tp)
    fp   = int(cm[:, class_idx].sum() - tp)
    rec  = tp / (tp + fn + 1e-9)
    prec = tp / (tp + fp + 1e-9)
    f1   = 2 * prec * rec / (prec + rec + 1e-9)
    return {"tp": tp, "fn": fn, "fp": fp,
            "recall": round(rec, 4),
            "precision": round(prec, 4),
            "f1": round(f1, 4)}


# ======================================================
# TWO-STAGE PREDICTION
# ======================================================

def two_stage_predict(s1, s2, X, stage1_thresh, stage2_thresh):
    s1_probs     = s1.predict(X, verbose=0)
    abnormal_mask = s1_probs[:, 1] >= stage1_thresh
    y_pred        = np.zeros(len(X), dtype=np.int32)
    if abnormal_mask.sum() > 0:
        s2_probs = s2.predict(X[abnormal_mask], verbose=0)
        y_pred[abnormal_mask] = np.where(s2_probs[:, 1] >= stage2_thresh, 2, 1)
    return y_pred


# ======================================================
# STAGE 2 THRESHOLD OPTIMISATION
# ======================================================

def optimize_stage2_threshold(s2_model, X_abn_val, _unused,
                               y_true_orig, abnormal_mask_val,
                               min_recall=0.95, min_precision=0.70):
    print(f"\n  Stage2 threshold sweep (recall≥{min_recall}, prec≥{min_precision}):")
    print(f"  {'Thresh':>8} {'Recall':>8} {'Prec':>8} {'F1':>8} {'TP':>4} {'FN':>4} {'FP':>4}")

    s2_probs = s2_model.predict(X_abn_val, verbose=0)
    results  = []
    for thresh in np.arange(0.05, 0.961, 0.025):
        abn_pred = np.where(s2_probs[:, 1] >= thresh, 2, 1)
        y_pred   = np.zeros(len(y_true_orig), dtype=np.int32)
        y_pred[abnormal_mask_val] = abn_pred
        cm = confusion_matrix(y_true_orig, y_pred, labels=[0, 1, 2])
        m  = per_class_metrics(cm, 2)
        print(f"  {thresh:>8.3f} {m['recall']:>8.3f} {m['precision']:>8.3f} "
              f"{m['f1']:>8.3f} {m['tp']:>4} {m['fn']:>4} {m['fp']:>4}")
        results.append((thresh, m["recall"], m["precision"], m["f1"]))

    strict  = [(t, r, p, f) for t, r, p, f in results
               if r >= min_recall and p >= min_precision]
    relaxed = [(t, r, p, f) for t, r, p, f in results if r >= min_recall]

    if strict:
        chosen = max(strict,   key=lambda x: x[3])
        label  = f"recall≥{min_recall} AND prec≥{min_precision}"
    elif relaxed:
        chosen = max(relaxed,  key=lambda x: x[3])
        label  = f"recall≥{min_recall} (precision relaxed)"
    else:
        chosen = max(results,  key=lambda x: (x[1], x[2]))
        label  = "best available recall"

    t, r, p, f = chosen
    print(f"\n  [Selected] Thresh={t:.3f} | Recall={r:.3f} | Prec={p:.3f} | F1={f:.3f}")
    print(f"  Strategy: {label}")
    return float(t)


# ======================================================
# TRAIN ONE FOLD  — returns models AND history objects
# ======================================================

def train_fold(X_tr, y_tr, X_val, y_val, verbose_fit=0):
    input_dim = X_tr.shape[1]

    # Stage 1
    y_tr_s1  = make_stage1_labels(y_tr)
    y_val_s1 = make_stage1_labels(y_val)
    print("  Oversampling Stage 1...")
    X_tr_s1_bal, y_tr_s1_bal = oversample(X_tr, y_tr_s1)
    cw1 = dict(enumerate(compute_class_weight(
        "balanced", classes=np.unique(y_tr_s1_bal), y=y_tr_s1_bal)))
    s1  = build_stage1(input_dim)
    h1  = s1.fit(X_tr_s1_bal, y_tr_s1_bal, epochs=100, batch_size=32,
                 validation_data=(X_val, y_val_s1),
                 callbacks=get_callbacks(), class_weight=cw1,
                 verbose=verbose_fit)

    # Stage 2
    X_tr_abn, y_tr_s2   = make_stage2_data(X_tr, y_tr)
    X_val_abn, y_val_s2 = make_stage2_data(X_val, y_val)
    print("  Oversampling Stage 2...")
    X_tr_s2_bal, y_tr_s2_bal = oversample(X_tr_abn, y_tr_s2)
    cw2 = dict(enumerate(compute_class_weight(
        "balanced", classes=np.unique(y_tr_s2_bal), y=y_tr_s2_bal)))
    s2  = build_stage2(input_dim)
    h2  = s2.fit(X_tr_s2_bal, y_tr_s2_bal, epochs=100, batch_size=16,
                 validation_data=(X_val_abn, y_val_s2),
                 callbacks=get_callbacks(), class_weight=cw2,
                 verbose=verbose_fit)

    return s1, s2, h1, h2


# ======================================================
# CROSS-VALIDATION
# ======================================================

def cross_validate(X, y):
    print(f"\n{'='*60}")
    print(f"Stratified {N_CV_FOLDS}-Fold Cross-Validation")
    print(f"{'='*60}")

    skf          = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True,
                                   random_state=RANDOM_SEED)
    fold_metrics = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\nFold {fold}/{N_CV_FOLDS}")
        X_tr, y_tr   = X[tr_idx], y[tr_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        sc    = StandardScaler()
        X_tr  = sc.fit_transform(X_tr)
        X_val = sc.transform(X_val)

        s1, s2, _, _ = train_fold(X_tr, y_tr, X_val, y_val)

        s1_val_probs = s1.predict(X_val, verbose=0)
        abn_mask_val = s1_val_probs[:, 1] >= STAGE1_THRESH
        X_abn_val    = X_val[abn_mask_val]

        stage2_thresh = (optimize_stage2_threshold(
            s2, X_abn_val, None, y_val, abn_mask_val)
            if abn_mask_val.sum() > 0 else 0.5)

        y_pred = two_stage_predict(s1, s2, X_val,
                                   STAGE1_THRESH, stage2_thresh)
        cm   = confusion_matrix(y_val, y_pred, labels=[0, 1, 2])
        acc  = np.mean(y_pred == y_val)
        sev  = per_class_metrics(cm, 2)
        mild = per_class_metrics(cm, 1)

        print(f"  Fold {fold}: acc={acc:.3f} | "
              f"sev_recall={sev['recall']:.3f} | "
              f"sev_prec={sev['precision']:.3f} | "
              f"mild_recall={mild['recall']:.3f}")

        fold_metrics.append({"accuracy": acc,
                             "sev_recall":    sev["recall"],
                             "sev_precision": sev["precision"],
                             "mild_recall":   mild["recall"]})

    df = pd.DataFrame(fold_metrics)
    print(f"\nCV Summary (mean ± std over {N_CV_FOLDS} folds):")
    for col in df.columns:
        print(f"  {col}: {df[col].mean():.3f} ± {df[col].std():.3f}")
    return df


# ======================================================
# INT8 TFLITE
# ======================================================

def convert_tflite(model, X_rep, name):
    def rep_gen():
        idx = np.random.choice(len(X_rep), size=min(300, len(X_rep)), replace=False)
        for i in idx:
            yield [X_rep[i:i+1].astype(np.float32)]

    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations             = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset    = rep_gen
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type      = tf.int8
    conv.inference_output_type     = tf.int8
    tflite_bytes = conv.convert()
    path = ARTIFACT_DIR / f"{name}_int8.tflite"
    path.write_bytes(tflite_bytes)
    size_kb = path.stat().st_size / 1024
    print(f"  {name}: {size_kb:.2f} KB → {path}")
    return path, size_kb


def tflite_predict_probs(tflite_path, X):
    interp = tf.lite.Interpreter(model_path=str(tflite_path))
    interp.allocate_tensors()
    in_d     = interp.get_input_details()[0]
    out_d    = interp.get_output_details()[0]
    sc, zp   = in_d["quantization"]
    o_sc,o_zp = out_d["quantization"]
    probs    = []
    for i in range(len(X)):
        xq = (X[i:i+1].astype(np.float32) / sc + zp).astype(np.int8)
        interp.set_tensor(in_d["index"], xq)
        interp.invoke()
        yq = interp.get_tensor(out_d["index"])
        probs.append((yq.astype(np.float32) - o_zp) * o_sc)
    return np.vstack(probs)


# ======================================================
# LATENCY
# ======================================================

def measure_latency(s1, s2, X_sample, n=1000):
    for _ in range(10):
        p = s1.predict(X_sample, verbose=0)
        if p[0, 1] >= STAGE1_THRESH:
            s2.predict(X_sample, verbose=0)
    t0 = time.time()
    for _ in range(n):
        p = s1.predict(X_sample, verbose=0)
        if p[0, 1] >= STAGE1_THRESH:
            s2.predict(X_sample, verbose=0)
    return (time.time() - t0) * 1000 / n


# ======================================================
# MAIN
# ======================================================

def main():
    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load
    X_raw, y_raw, X_test, y_test, _ = load_data()

    # Figure 1 — class distribution
    print("\nGenerating Figure 1: class distribution...")
    plot_class_distribution(y_raw, y_test)

    # 2. Cross-validation
    cv_results = cross_validate(X_raw, y_raw)

    # Figure 5 — CV results
    print("\nGenerating Figure 5: CV fold performance...")
    plot_cv_results(cv_results)

    # 3. Final model
    print(f"\n{'='*60}")
    print("Final Model Training")
    print(f"{'='*60}")

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_raw, y_raw, test_size=VAL_FRACTION,
        stratify=y_raw, random_state=RANDOM_SEED)

    scaler    = StandardScaler()
    X_tr      = scaler.fit_transform(X_tr)
    X_val     = scaler.transform(X_val)
    X_test_sc = scaler.transform(X_test)

    s1, s2, h1, h2 = train_fold(X_tr, y_tr, X_val, y_val, verbose_fit=1)

    # Figures 2 & 3 — training curves
    print("\nGenerating Figure 2: Stage 1 training curves...")
    plot_training_curves(h1, "Stage 1 — Normal vs Abnormal", "fig2_stage1_training_curves.png")
    print("Generating Figure 3: Stage 2 training curves...")
    plot_training_curves(h2, "Stage 2 — Mild vs Severe",     "fig3_stage2_training_curves.png")

    # 4. Threshold selection on val set
    s1_val_probs = s1.predict(X_val, verbose=0)
    abn_mask_val = s1_val_probs[:, 1] >= STAGE1_THRESH
    X_abn_val    = X_val[abn_mask_val]

    stage2_thresh = (optimize_stage2_threshold(
        s2, X_abn_val, None, y_val, abn_mask_val)
        if abn_mask_val.sum() > 0 else 0.5)

    # 5. Evaluate
    print(f"\n{'='*60}")
    print("Evaluation on held-out test set")
    print(f"{'='*60}")

    y_pred = two_stage_predict(s1, s2, X_test_sc, STAGE1_THRESH, stage2_thresh)
    cm     = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
    acc    = np.mean(y_pred == y_test)

    print(f"\nStage1 threshold: {STAGE1_THRESH}  |  Stage2 threshold: {stage2_thresh:.3f}")
    print("\nClassification Report (test set):")
    print(classification_report(y_test, y_pred,
          target_names=["Normal (0)", "Mild (1)", "Severe (2)"], zero_division=0))
    print("Confusion Matrix:")
    print(cm)

    sev_m    = per_class_metrics(cm, 2)
    mild_m   = per_class_metrics(cm, 1)
    norm_m   = per_class_metrics(cm, 0)
    macro_f1 = float(np.mean([per_class_metrics(cm, i)["f1"] for i in range(3)]))

    # 6. Latency
    latency_ms = measure_latency(s1, s2, X_test_sc[:1])

    # 7. TFLite conversion
    print("\nConverting to INT8 TFLite...")
    p1, kb1      = convert_tflite(s1, X_tr, "stage1_normal_vs_abnormal")
    X_tr_abn, _  = make_stage2_data(X_tr, y_tr)
    p2, kb2      = convert_tflite(s2, X_tr_abn, "stage2_mild_vs_severe")
    total_kb     = kb1 + kb2

    # 8. Validate INT8
    print("\nValidating INT8 two-stage pipeline...")
    s1_tflite_probs = tflite_predict_probs(p1, X_test_sc)
    abn_mask_test   = s1_tflite_probs[:, 1] >= STAGE1_THRESH
    y_tflite        = np.zeros(len(X_test_sc), dtype=np.int32)
    if abn_mask_test.sum() > 0:
        s2_tflite_probs = tflite_predict_probs(p2, X_test_sc[abn_mask_test])
        y_tflite[abn_mask_test] = np.where(
            s2_tflite_probs[:, 1] >= stage2_thresh, 2, 1)

    tflite_cm  = confusion_matrix(y_test, y_tflite, labels=[0, 1, 2])
    tflite_acc = np.mean(y_tflite == y_test)
    tflite_sev = per_class_metrics(tflite_cm, 2)

    print("\nINT8 TFLite Classification Report (test set):")
    print(classification_report(y_test, y_tflite,
          target_names=["Normal (0)", "Mild (1)", "Severe (2)"], zero_division=0))

    # Figures 4, 6, 7
    print("\nGenerating Figure 4: confusion matrices...")
    plot_confusion_matrices(cm, tflite_cm, y_test)

    print("Generating Figure 6: threshold sweep...")
    s1_test_probs   = s1.predict(X_test_sc, verbose=0)
    abn_mask_test_f = s1_test_probs[:, 1] >= STAGE1_THRESH
    if abn_mask_test_f.sum() > 0:
        plot_threshold_sweep(y_test, s2, X_test_sc[abn_mask_test_f],
                             abn_mask_test_f, stage2_thresh)

    results_dict = {
        "accuracy":      acc,
        "sev_recall":    sev_m["recall"],
        "sev_precision": sev_m["precision"],
        "total_kb":      total_kb,
        "latency_ms":    latency_ms,
    }
    print("Generating Figure 7: NFR compliance...")
    plot_nfr_compliance(results_dict)

    # 9. Final summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")

    print(f"\n  Cross-validation ({N_CV_FOLDS} folds on train set):")
    for col in cv_results.columns:
        print(f"    {col}: {cv_results[col].mean():.3f} ± {cv_results[col].std():.3f}")

    print(f"\n  Held-out test set (Float32, two-stage):")
    print(f"    Overall accuracy:  {acc:.4f}")
    print(f"    Macro F1:          {macro_f1:.4f}")
    print(f"    Severe recall:     {sev_m['recall']:.4f}")
    print(f"    Severe precision:  {sev_m['precision']:.4f}")
    print(f"    Mild recall:       {mild_m['recall']:.4f}")
    print(f"    Normal recall:     {norm_m['recall']:.4f}")

    print(f"\n  INT8 TFLite (two-stage combined):")
    print(f"    Accuracy:          {tflite_acc:.4f}")
    print(f"    Severe recall:     {tflite_sev['recall']:.4f}")
    print(f"    Severe precision:  {tflite_sev['precision']:.4f}")
    print(f"    Stage1 size:       {kb1:.2f} KB")
    print(f"    Stage2 size:       {kb2:.2f} KB")
    print(f"    Total size:        {total_kb:.2f} KB")
    print(f"    Combined latency:  {latency_ms:.3f} ms")

    print(f"\n  NFR Compliance:")
    checks = {
        "Accuracy ≥ 92%":         (acc >= 0.92,                f"{acc*100:.1f}%"),
        "Severe recall ≥ 95%":    (sev_m['recall'] >= 0.95,    f"{sev_m['recall']*100:.1f}%"),
        "Severe precision ≥ 90%": (sev_m['precision'] >= 0.90, f"{sev_m['precision']*100:.1f}%"),
        "Total size ≤ 100 KB":    (total_kb <= 100,             f"{total_kb:.2f} KB"),
        "Latency ≤ 50 ms":        (latency_ms <= 50,            f"{latency_ms:.1f} ms"),
    }
    for name, (passes, value) in checks.items():
        print(f"    {name}: {value}  {'✓' if passes else '✗ (not met)'}")

    print(f"\n  Plots saved to: {PLOT_DIR}")
    print("\nTraining complete.")

    return {**results_dict,
            "cv": cv_results, "macro_f1": macro_f1,
            "mild_recall": mild_m["recall"],
            "stage1_thresh": STAGE1_THRESH,
            "stage2_thresh": stage2_thresh,
            "int8_sev_recall": tflite_sev["recall"]}


if __name__ == "__main__":
    main()