"""Execute Phases 1-9 with validation outputs."""
from __future__ import annotations

import json
from pathlib import Path

from training.data_pipeline import main as phase1
from training.train_tinyml_model import main as phase2_3
from simulation.run_benchmark import main as phase8_9


def validate_constraints() -> dict:
    m23 = json.loads(Path("training/artifacts/phase2_3_metrics.json").read_text())
    sim = json.loads(Path("simulation/artifacts/benchmark_results.json").read_text())
    checks = {
        "accuracy>=0.92": m23["int8_accuracy"] >= 0.92,
        "edge_latency<=50ms": m23["edge_inference_latency_ms"] <= 50,
        "sram<=100kb": m23["estimated_sram_kb"] <= 100,
        "flash<=500kb": m23["estimated_flash_kb"] <= 500,
        "e2e_latency<=200": sim["edge_fog_cloud"]["latency_ms"] <= 200,
        "bandwidth_reduction>=80": sim["bandwidth_reduction_pct"] >= 80,
        "severe_recall>=95": sim["severe_recall"] >= 0.95,
        "severe_precision>=90": sim["severe_precision"] >= 0.90,
        "false_positives<=5": sim["false_positive_rate"] <= 0.05,
    }
    Path("simulation/artifacts/final_validation.json").write_text(json.dumps(checks, indent=2))
    return checks


def main() -> None:
    print("Phase 1 - Data pipeline")
    phase1()
    print("Phase 2/3 - Training + quantization")
    phase2_3()
    print("Phase 8/9 - Benchmark")
    phase8_9()
    checks = validate_constraints()
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
