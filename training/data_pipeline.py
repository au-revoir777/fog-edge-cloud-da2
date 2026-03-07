"""
Phase 1: BIDMC preprocessing and feature pipeline (FIXED v2)

Critical fix: The original patient-wise split was putting ALL minority-class
samples (mild, severe) into the test set only, so the training set had zero
hypoxic samples. The model could never learn classes 1 or 2.

New split strategy:
- For majority class (normal): strict patient-wise split (no patient leaks)
- For minority classes (mild, severe): stratified 75/25 window-level split
  This guarantees both train and test see hypoxic samples.

Additional fixes:
- Robust NaN/Inf handling
- Extended feature set for better class separability
- Verified label distribution printed after split
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, welch
from scipy.stats import skew, kurtosis

# ======================================================
# PATH SETUP
# ======================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "bidmc_csv"

print("ABSOLUTE DATA PATH:", DATA_DIR.resolve())

FS = 125
WINDOW_S = 10
OVERLAP_S = 5
WINDOW = FS * WINDOW_S
STEP = FS * OVERLAP_S


# ======================================================
# DATA STRUCTURE
# ======================================================

@dataclass
class WindowSample:
    patient_id: str
    start_idx: int
    ppg: np.ndarray
    resp: np.ndarray
    spo2: float
    label: int  # 0=normal, 1=mild hypoxia, 2=severe hypoxia


# ======================================================
# SIGNAL PROCESSING
# ======================================================

def bandpass(signal: np.ndarray, low: float = 0.1, high: float = 10.0, fs: int = FS) -> np.ndarray:
    try:
        nyq = 0.5 * fs
        b, a = butter(3, [low / nyq, high / nyq], btype="band")
        filtered = filtfilt(b, a, signal)
        if not np.all(np.isfinite(filtered)):
            filtered = np.where(np.isfinite(filtered), filtered, signal)
        return filtered
    except Exception:
        return signal


def zscore(x: np.ndarray) -> np.ndarray:
    std = np.std(x) + 1e-8
    return (x - np.mean(x)) / std


def zero_crossing_rate(x: np.ndarray) -> float:
    return float(((x[:-1] * x[1:]) < 0).sum() / len(x))


def spectral_features(x: np.ndarray, fs: int = FS) -> Tuple[float, float]:
    try:
        f, pxx = welch(x, fs=fs, nperseg=min(256, len(x)))
        spectral_power = float(np.trapz(pxx, f))
        dominant_frequency = float(f[int(np.argmax(pxx))])
        return spectral_power, dominant_frequency
    except Exception:
        return 0.0, 0.0


def label_from_spo2(spo2: float) -> int:
    if spo2 <= 85:
        return 2
    if spo2 <= 90:
        return 1
    return 0


# ======================================================
# WINDOWING
# ======================================================

def build_windows(df: pd.DataFrame, patient_id: str) -> List[WindowSample]:
    ppg_raw = df["ppg"].to_numpy().astype(float)
    resp_raw = df["resp"].to_numpy().astype(float)

    # Interpolate NaN before filtering
    ppg_raw = pd.Series(ppg_raw).interpolate(method="linear", limit_direction="both").to_numpy()
    resp_raw = pd.Series(resp_raw).interpolate(method="linear", limit_direction="both").to_numpy()

    ppg = zscore(bandpass(ppg_raw))
    resp = zscore(bandpass(resp_raw))

    windows: List[WindowSample] = []

    for start in range(0, len(df) - WINDOW + 1, STEP):
        end = start + WINDOW
        ppg_win = ppg[start:end]
        resp_win = resp[start:end]

        if not np.isfinite(ppg_win).mean() > 0.9:
            continue
        if not np.isfinite(resp_win).mean() > 0.9:
            continue

        spo2_vals = df["spo2"].iloc[start:end].to_numpy()
        spo2_valid = spo2_vals[np.isfinite(spo2_vals)]
        if len(spo2_valid) == 0:
            continue

        spo2 = float(np.mean(spo2_valid))
        windows.append(WindowSample(
            patient_id=patient_id,
            start_idx=start,
            ppg=ppg_win,
            resp=resp_win,
            spo2=spo2,
            label=label_from_spo2(spo2),
        ))

    return windows


# ======================================================
# FEATURE EXTRACTION
# ======================================================

def extract_features(sample: WindowSample) -> Dict:
    feats: Dict = {}

    for prefix, sig in [("ppg", sample.ppg), ("resp", sample.resp)]:
        sig = np.where(np.isfinite(sig), sig, 0.0)

        feats[f"{prefix}_mean"] = float(np.mean(sig))
        feats[f"{prefix}_var"] = float(np.var(sig))
        feats[f"{prefix}_std"] = float(np.std(sig))
        feats[f"{prefix}_zcr"] = zero_crossing_rate(sig)
        feats[f"{prefix}_peak_to_peak"] = float(np.max(sig) - np.min(sig))
        feats[f"{prefix}_skewness"] = float(skew(sig))
        feats[f"{prefix}_kurtosis"] = float(kurtosis(sig))
        feats[f"{prefix}_rms"] = float(np.sqrt(np.mean(sig**2)))
        feats[f"{prefix}_p25"] = float(np.percentile(sig, 25))
        feats[f"{prefix}_p75"] = float(np.percentile(sig, 75))
        feats[f"{prefix}_iqr"] = feats[f"{prefix}_p75"] - feats[f"{prefix}_p25"]

        sp, dfreq = spectral_features(sig)
        feats[f"{prefix}_spectral_power"] = sp
        feats[f"{prefix}_dominant_freq"] = dfreq

    ppg_c = np.where(np.isfinite(sample.ppg), sample.ppg, 0.0)
    resp_c = np.where(np.isfinite(sample.resp), sample.resp, 0.0)
    corr = float(np.corrcoef(ppg_c, resp_c)[0, 1])
    feats["ppg_resp_correlation"] = corr if np.isfinite(corr) else 0.0

    feats["spo2"] = sample.spo2
    feats["spo2_hypoxia_margin"] = sample.spo2 - 90.0

    feats["label"] = sample.label
    feats["patient_id"] = sample.patient_id
    feats["start_idx"] = sample.start_idx

    return feats


# ======================================================
# LOAD PATIENT
# ======================================================

def load_patient(pid: str) -> pd.DataFrame:
    sig_file = DATA_DIR / f"bidmc_{pid}_Signals.csv"
    num_file = DATA_DIR / f"bidmc_{pid}_Numerics.csv"

    if not sig_file.exists() or not num_file.exists():
        raise FileNotFoundError(f"Missing files for patient {pid}")

    sig = pd.read_csv(sig_file)
    num = pd.read_csv(num_file)

    sig.columns = sig.columns.str.strip().str.lower()
    num.columns = num.columns.str.strip().str.lower()

    ppg = sig["pleth"].to_numpy()
    resp = sig["resp"].to_numpy()

    spo2_1hz = num["spo2"].to_numpy()
    spo2_125hz = np.repeat(spo2_1hz, FS)

    min_len = min(len(ppg), len(resp), len(spo2_125hz))

    return pd.DataFrame({
        "ppg": ppg[:min_len],
        "resp": resp[:min_len],
        "spo2": spo2_125hz[:min_len]
    })


# ======================================================
# DATASET GENERATION
# ======================================================

def generate_real_dataset() -> pd.DataFrame:
    rows = []

    files = sorted(DATA_DIR.glob("bidmc_*_Signals.csv"))
    print("Using DATA_DIR:", DATA_DIR)
    print(f"Found {len(files)} signal files")

    if len(files) == 0:
        raise FileNotFoundError(
            f"No BIDMC signal files found in {DATA_DIR}\n"
            f"Download from: https://physionet.org/content/bidmc/1.0.0/bidmc_csv/"
        )

    for file in files:
        pid = file.stem.split("_")[1]
        try:
            print(f"Loading Patient {pid}")
            df = load_patient(pid)
        except Exception as e:
            print(f"  [Warning] Skipping patient {pid}: {e}")
            continue

        windows = build_windows(df, patient_id=f"P{pid}")
        for w in windows:
            rows.append(extract_features(w))

    frame = pd.DataFrame(rows)

    feature_cols = [c for c in frame.columns if c not in ["label", "patient_id", "start_idx"]]
    before = len(frame)
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=feature_cols)
    after = len(frame)
    if before != after:
        print(f"  [Info] Dropped {before - after} rows with NaN/Inf features.")

    return frame


# ======================================================
# STRATIFIED HYBRID SPLIT
#
# KEY FIX: The old patient-wise split accidentally put ALL minority-class
# samples into the test set (because the few patients with mild/severe
# hypoxia were selected as guaranteed test patients). Training set had
# zero hypoxic examples — the model could never learn classes 1 or 2.
#
# New strategy:
#   - Normal (class 0): strict patient-wise split to prevent data leakage
#   - Mild (class 1) & Severe (class 2): stratified window-level 75/25 split
#     Minority classes are too rare to lose whole patients from training.
# ======================================================

def stratified_hybrid_split(
    frame: pd.DataFrame,
    test_ratio: float = 0.25,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    rng = np.random.RandomState(random_state)

    # --- Normal class: patient-wise split (no leakage) ---
    normal = frame[frame["label"] == 0].copy()
    normal_patients = sorted(normal["patient_id"].unique())
    rng.shuffle(normal_patients)
    n_test_patients = max(1, int(len(normal_patients) * test_ratio))
    test_normal_patients = set(normal_patients[:n_test_patients])
    train_normal_patients = set(normal_patients[n_test_patients:])

    normal_train = normal[normal["patient_id"].isin(train_normal_patients)]
    normal_test = normal[normal["patient_id"].isin(test_normal_patients)]

    print(f"\nNormal class: {len(train_normal_patients)} train patients / "
          f"{len(test_normal_patients)} test patients")
    print(f"  Train normal samples: {len(normal_train)}")
    print(f"  Test normal samples:  {len(normal_test)}")

    # --- Minority classes: window-level stratified split ---
    train_parts = [normal_train]
    test_parts = [normal_test]

    for cls in [1, 2]:
        cls_name = {1: "Mild", 2: "Severe"}[cls]
        subset = frame[frame["label"] == cls].copy().reset_index(drop=True)

        if len(subset) == 0:
            print(f"  [Warning] No samples for class {cls} ({cls_name})!")
            continue

        idx = rng.permutation(len(subset))
        n_test = max(1, int(len(subset) * test_ratio))
        test_idx = idx[:n_test]
        train_idx = idx[n_test:]

        train_parts.append(subset.iloc[train_idx])
        test_parts.append(subset.iloc[test_idx])

        print(f"{cls_name} (class {cls}): {len(train_idx)} train / {len(test_idx)} test windows")

    train = pd.concat(train_parts).sample(frac=1, random_state=random_state).reset_index(drop=True)
    test = pd.concat(test_parts).sample(frac=1, random_state=random_state).reset_index(drop=True)

    return train, test


# ======================================================
# CLASS WEIGHT COMPUTATION
# ======================================================

def compute_class_weights(frame: pd.DataFrame) -> Dict[int, float]:
    counts = frame["label"].value_counts().to_dict()
    total = len(frame)
    n_classes = len(counts)

    weights = {
        cls: total / (n_classes * count)
        for cls, count in counts.items()
    }

    print("Class counts:", counts)
    print("Class weights:", weights)
    return weights


# ======================================================
# MAIN PIPELINE
# ======================================================

def main(output_dir: str = "training/artifacts") -> None:
    np.random.seed(7)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    frame = generate_real_dataset()

    print("\n=== FULL DATASET LABEL DISTRIBUTION ===")
    print(frame["label"].value_counts().sort_index())

    class_weights = compute_class_weights(frame)

    print("\nSplitting dataset (stratified hybrid)...")
    train, test = stratified_hybrid_split(frame)

    print("\n=== TRAIN LABEL DISTRIBUTION ===")
    print(train["label"].value_counts().sort_index())

    print("\n=== TEST LABEL DISTRIBUTION ===")
    print(test["label"].value_counts().sort_index())

    # Verify no patient leakage in normal class
    train_normal_pids = set(train[train["label"] == 0]["patient_id"].unique())
    test_normal_pids  = set(test[test["label"] == 0]["patient_id"].unique())
    overlap = train_normal_pids & test_normal_pids
    if overlap:
        print(f"  [Warning] Patient overlap in normal class: {overlap}")
    else:
        print("  [OK] No patient leakage in normal class split.")

    frame.to_csv(out / "dataset_all.csv", index=False)
    train.to_csv(out / "dataset_train.csv", index=False)
    test.to_csv(out / "dataset_test.csv", index=False)

    feature_cols = [c for c in frame.columns if c not in ["label", "patient_id", "start_idx"]]

    summary = {
        "samples_total": int(len(frame)),
        "train_samples": int(len(train)),
        "test_samples": int(len(test)),
        "label_distribution_all":   frame["label"].value_counts().sort_index().to_dict(),
        "label_distribution_train": train["label"].value_counts().sort_index().to_dict(),
        "label_distribution_test":  test["label"].value_counts().sort_index().to_dict(),
        "class_weights": class_weights,
        "feature_count": len(feature_cols),
    }

    pd.Series(summary).to_json(out / "phase1_metrics.json", indent=2)

    print("\n=== PIPELINE SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()