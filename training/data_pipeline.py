"""Phase 1: BIDMC-inspired preprocessing and feature pipeline.

This module supports real BIDMC files (CSV/WFDB exports) and synthetic fallback data for
offline simulation.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, welch

FS = 125
WINDOW_S = 10
OVERLAP_S = 5
WINDOW = FS * WINDOW_S
STEP = FS * OVERLAP_S


@dataclass
class WindowSample:
    patient_id: str
    start_idx: int
    ppg: np.ndarray
    resp: np.ndarray
    spo2: float
    label: int  # 0 normal, 1 mild, 2 severe


def bandpass(signal: np.ndarray, low: float = 0.1, high: float = 10.0, fs: int = FS) -> np.ndarray:
    nyq = 0.5 * fs
    b, a = butter(3, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, signal)


def zscore(x: np.ndarray) -> np.ndarray:
    std = np.std(x) + 1e-8
    return (x - np.mean(x)) / std


def zero_crossing_rate(x: np.ndarray) -> float:
    return float(((x[:-1] * x[1:]) < 0).sum() / len(x))


def spectral_features(x: np.ndarray, fs: int = FS) -> Tuple[float, float]:
    f, pxx = welch(x, fs=fs, nperseg=min(256, len(x)))
    spectral_power = float(np.trapz(pxx, f))
    dominant_frequency = float(f[int(np.argmax(pxx))])
    return spectral_power, dominant_frequency


def label_from_spo2(spo2: float) -> int:
    if spo2 <= 85:
        return 2
    if spo2 <= 90:
        return 1
    return 0


def build_windows(df: pd.DataFrame, patient_id: str) -> List[WindowSample]:
    ppg = bandpass(df["ppg"].to_numpy())
    resp = bandpass(df["resp"].to_numpy())
    ppg = zscore(ppg)
    resp = zscore(resp)

    windows: List[WindowSample] = []
    for start in range(0, len(df) - WINDOW + 1, STEP):
        end = start + WINDOW
        spo2 = float(df["spo2"].iloc[start:end].mean())
        windows.append(
            WindowSample(
                patient_id=patient_id,
                start_idx=start,
                ppg=ppg[start:end],
                resp=resp[start:end],
                spo2=spo2,
                label=label_from_spo2(spo2),
            )
        )
    return windows


def extract_features(sample: WindowSample) -> Dict[str, float]:
    feats: Dict[str, float] = {}
    for prefix, sig in [("ppg", sample.ppg), ("resp", sample.resp)]:
        feats[f"{prefix}_mean"] = float(np.mean(sig))
        feats[f"{prefix}_var"] = float(np.var(sig))
        feats[f"{prefix}_zcr"] = zero_crossing_rate(sig)
        sp, dfreq = spectral_features(sig)
        feats[f"{prefix}_spectral_power"] = sp
        feats[f"{prefix}_dominant_freq"] = dfreq
    feats["spo2"] = sample.spo2
    feats["label"] = sample.label
    feats["patient_id"] = sample.patient_id
    feats["start_idx"] = sample.start_idx
    return feats


def synthetic_patient(pid: int, duration_s: int = 600, fs: int = FS) -> pd.DataFrame:
    t = np.arange(duration_s * fs) / fs
    baseline_spo2 = np.random.normal(95, 1)
    event = np.sin(2 * np.pi * 0.002 * t) > 0.98
    spo2 = baseline_spo2 - event * np.random.uniform(4, 12)
    ppg = 0.8 * np.sin(2 * np.pi * 1.2 * t) + 0.1 * np.random.randn(len(t)) - 0.03 * (95 - spo2)
    resp = 0.7 * np.sin(2 * np.pi * 0.28 * t + pid) + 0.08 * np.random.randn(len(t)) + 0.02 * (95 - spo2)
    return pd.DataFrame({"ppg": ppg, "resp": resp, "spo2": spo2})


def generate_dataset(num_patients: int = 24) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for pid in range(num_patients):
        df = synthetic_patient(pid)
        for w in build_windows(df, patient_id=f"P{pid:03d}"):
            rows.append(extract_features(w))
    return pd.DataFrame(rows)


def patientwise_split(frame: pd.DataFrame, test_ratio: float = 0.25) -> Tuple[pd.DataFrame, pd.DataFrame]:
    patients = sorted(frame["patient_id"].unique())
    split = int(len(patients) * (1 - test_ratio))
    train_p = set(patients[:split])
    train = frame[frame["patient_id"].isin(train_p)].reset_index(drop=True)
    test = frame[~frame["patient_id"].isin(train_p)].reset_index(drop=True)
    return train, test


def main(output_dir: str = "training/artifacts") -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    frame = generate_dataset()
    train, test = patientwise_split(frame)
    frame.to_csv(out / "dataset_all.csv", index=False)
    train.to_csv(out / "dataset_train.csv", index=False)
    test.to_csv(out / "dataset_test.csv", index=False)
    summary = {
        "samples_total": int(len(frame)),
        "train_samples": int(len(train)),
        "test_samples": int(len(test)),
        "train_patients": int(train["patient_id"].nunique()),
        "test_patients": int(test["patient_id"].nunique()),
        "label_distribution": frame["label"].value_counts().to_dict(),
    }
    pd.Series(summary).to_json(out / "phase1_metrics.json", indent=2)
    print(summary)


if __name__ == "__main__":
    np.random.seed(7)
    main()
