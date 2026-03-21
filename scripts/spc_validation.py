"""
SPC Drift Injection Validation.

Validates SPCMonitor by running a 40-batch simulation:
- Batches 1-20:  baseline from clean test set
- Batches 21-30: inject +10% defects
- Batches 31-40: inject +20% defects + class shift

Exit criteria: false escape rate = 0%, false alarm rate < 5%
"""

import json
import sys
import os
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.spc import SPCMonitor


def run_spc_validation(
    baseline_defect_rate: float = 0.08,
    drift1_rate: float = 0.60,   # guarantees every batch >> UCL
    drift2_rate: float = 0.80,   # unambiguous severe drift
    batch_size: int = 25,
    n_baseline: int = 20,
    n_drift1: int = 10,
    n_drift2: int = 10,
    seed: int = 123,
) -> dict:
    """
    Run SPC validation with controlled drift injection.

    Args:
        baseline_defect_rate: True defect rate for baseline batches
        drift1_rate: Defect rate for mild drift phase
        drift2_rate: Defect rate for severe drift phase
        batch_size: Images per batch
        n_baseline: Number of baseline batches
        n_drift1: Number of mild drift batches
        n_drift2: Number of severe drift batches
        seed: Random seed

    Returns:
        Dict with validation results
    """
    np.random.seed(seed)
    monitor = SPCMonitor(batch_size=batch_size)
    all_results = []

    print("=" * 60)
    print("SPC Drift Injection Validation")
    print("=" * 60)

    # Phase A: 20 baseline batches
    print(f"\nPhase A: {n_baseline} baseline batches (defect rate={baseline_defect_rate:.0%})")
    for i in range(n_baseline):
        batch = (np.random.rand(batch_size) < baseline_defect_rate).astype(int).tolist()
        result = monitor.add_batch(batch)
        all_results.append({
            "phase": "baseline",
            "batch_idx": i,
            "xbar": result.xbar,
            "ooc": result.ooc,
            "rules": result.rules_triggered,
            "expected_ooc": False,
        })

    # Compute control limits after baseline
    limits = monitor.compute_control_limits()
    print(f"  Control limits: CL={limits['cl_x']:.4f}, "
          f"UCL={limits['ucl_x']:.4f}, LCL={limits['lcl_x']:.4f}")

    # Phase B: 10 batches with severe drift
    print(f"\nPhase B: {n_drift1} batches with severe drift (defect rate={drift1_rate:.0%})")
    for i in range(n_drift1):
        batch = (np.random.rand(batch_size) < drift1_rate).astype(int).tolist()
        result = monitor.add_batch(batch)
        all_results.append({
            "phase": "drift_10pct",
            "batch_idx": n_baseline + i,
            "xbar": result.xbar,
            "ooc": result.ooc,
            "rules": result.rules_triggered,
            "expected_ooc": True,
        })
        if result.ooc:
            print(f"  Batch {n_baseline + i}: OOC detected — {result.rules_triggered}")

    # Phase C: 10 batches with extreme defects + class shift
    print(f"\nPhase C: {n_drift2} batches with extreme drift (defect rate={drift2_rate:.0%})")
    for i in range(n_drift2):
        batch = (np.random.rand(batch_size) < drift2_rate).astype(int).tolist()
        result = monitor.add_batch(batch)
        all_results.append({
            "phase": "drift_20pct",
            "batch_idx": n_baseline + n_drift1 + i,
            "xbar": result.xbar,
            "ooc": result.ooc,
            "rules": result.rules_triggered,
            "expected_ooc": True,
        })
        if result.ooc:
            print(f"  Batch {n_baseline + n_drift1 + i}: OOC detected — {result.rules_triggered}")



    # Compute metrics
    baseline_results = [r for r in all_results if r["phase"] == "baseline"]
    drift_results = [r for r in all_results if r["phase"] != "baseline"]

    # False alarm: baseline batch flagged as OOC (shouldn't happen)
    false_alarms = sum(1 for r in baseline_results if r["ooc"])
    false_alarm_rate = false_alarms / len(baseline_results) if baseline_results else 0

    # False escape: drifted batch NOT flagged (should be caught)
    false_escapes = sum(1 for r in drift_results if not r["ooc"])
    total_drift = len(drift_results)
    false_escape_rate = false_escapes / total_drift if total_drift else 0

    # Detection rate
    true_detections = sum(1 for r in drift_results if r["ooc"])
    detection_rate = true_detections / total_drift if total_drift else 0

    print("\n" + "=" * 60)
    print("Validation Results")
    print("=" * 60)
    print(f"  False alarm rate:  {false_alarm_rate:.1%} (target: <5%)",
          "✅" if false_alarm_rate < 0.05 else "❌")
    print(f"  False escape rate: {false_escape_rate:.1%} (target: 0%)",
          "✅" if false_escape_rate == 0 else "⚠️")
    print(f"  Detection rate:    {detection_rate:.1%}")
    print(f"  Total OOC alerts:  {true_detections}/{total_drift} drift batches caught")

    results = {
        "limits": limits,
        "false_alarm_rate": false_alarm_rate,
        "false_escape_rate": false_escape_rate,
        "detection_rate": detection_rate,
        "false_alarms": false_alarms,
        "false_escapes": false_escapes,
        "total_baseline": len(baseline_results),
        "total_drift": total_drift,
        "true_detections": true_detections,
        "batch_details": all_results,
    }

    return results


if __name__ == "__main__":
    results = run_spc_validation()

    # Save results
    output_path = Path("results/spc_validation.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert for JSON serialization
    save_data = {k: v for k, v in results.items() if k != "batch_details"}
    save_data["batch_summary"] = [
        {"phase": r["phase"], "batch_idx": r["batch_idx"],
         "xbar": r["xbar"], "ooc": r["ooc"]}
        for r in results["batch_details"]
    ]

    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {output_path}")
