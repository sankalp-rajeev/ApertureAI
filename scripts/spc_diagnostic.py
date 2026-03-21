"""Diagnostic: print xbar vs UCL for every drift batch."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.spc import SPCMonitor
import numpy as np

np.random.seed(42)
monitor = SPCMonitor(batch_size=25)

# Phase A: 20 baseline batches (8% defect rate)
for _ in range(20):
    batch = (np.random.rand(25) < 0.08).astype(float)
    monitor.add_batch(batch)

limits = monitor.compute_control_limits()
print(f"Baseline: CL={limits['cl_x']:.4f}, UCL={limits['ucl_x']:.4f}, "
      f"LCL={limits['lcl_x']:.4f}, rbar={limits['rbar']:.4f}")
print()

# Phase B: drift +5% (total rate=13%)
print("=== Phase B: +5% drift (target rate=13%) ===")
for i in range(10):
    batch = (np.random.rand(25) < 0.13).astype(float)
    result = monitor.add_batch(batch)
    print(f"Batch {i+21}: xbar={result.xbar:.3f}, UCL={monitor.ucl_x:.3f}, "
          f"above_UCL={result.xbar > monitor.ucl_x}, OOC={result.ooc}, "
          f"rules={result.rules_triggered}")

print()
# Phase C: drift +15% (total rate=23%)
print("=== Phase C: +15% drift (target rate=23%) ===")
for i in range(10):
    batch = (np.random.rand(25) < 0.23).astype(float)
    result = monitor.add_batch(batch)
    print(f"Batch {i+31}: xbar={result.xbar:.3f}, UCL={monitor.ucl_x:.3f}, "
          f"above_UCL={result.xbar > monitor.ucl_x}, OOC={result.ooc}, "
          f"rules={result.rules_triggered}")
