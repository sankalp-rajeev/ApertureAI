"""
CPU Latency Benchmark: PyTorch vs ONNX Runtime vs OpenVINO

Task 2.6-2.7 — Three-way latency comparison on local Intel CPU.

Run this LOCALLY (not on Colab) to get accurate numbers for your hardware.

Usage:
    python scripts/benchmark_latency.py

Prerequisites:
    pip install onnxruntime openvino

Expected output fills Resume Bullet B1:
    - ONNX Runtime: <8ms target
    - OpenVINO: <5ms target
    - Speedup: ≥2x over PyTorch
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def benchmark_pytorch(model, input_tensor, num_runs=1000, warmup=100):
    """Benchmark PyTorch CPU inference."""
    model.eval()
    model.cpu()
    input_cpu = input_tensor.cpu()

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_cpu)

    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = model(input_cpu)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms

    return {
        "median_ms": np.median(times),
        "mean_ms": np.mean(times),
        "std_ms": np.std(times),
        "min_ms": np.min(times),
        "max_ms": np.max(times),
    }


def benchmark_onnx(onnx_path, input_array, num_runs=1000, warmup=100):
    """Benchmark ONNX Runtime CPU inference."""
    import onnxruntime as ort

    # Create session with CPU provider
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        str(onnx_path),
        sess_options,
        providers=["CPUExecutionProvider"]
    )

    input_name = session.get_inputs()[0].name

    # Warmup
    for _ in range(warmup):
        _ = session.run(None, {input_name: input_array})

    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = session.run(None, {input_name: input_array})
        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms

    return {
        "median_ms": np.median(times),
        "mean_ms": np.mean(times),
        "std_ms": np.std(times),
        "min_ms": np.min(times),
        "max_ms": np.max(times),
    }


def benchmark_openvino(xml_path, input_array, num_runs=1000, warmup=100):
    """Benchmark OpenVINO CPU inference."""
    import openvino as ov

    core = ov.Core()
    model = core.read_model(str(xml_path))
    compiled_model = core.compile_model(model, "CPU")
    infer_request = compiled_model.create_infer_request()

    # Get input tensor
    input_tensor = ov.Tensor(input_array)

    # Warmup
    for _ in range(warmup):
        infer_request.infer([input_tensor])

    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        infer_request.infer([input_tensor])
        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms

    return {
        "median_ms": np.median(times),
        "mean_ms": np.mean(times),
        "std_ms": np.std(times),
        "min_ms": np.min(times),
        "max_ms": np.max(times),
    }


def main():
    print("=" * 60)
    print("CPU Latency Benchmark")
    print("=" * 60)

    # Paths - update these if your models are elsewhere
    models_dir = PROJECT_ROOT / "models"
    pytorch_path = models_dir / "best_model.pth"
    onnx_path = models_dir / "model.onnx"
    openvino_path = models_dir / "openvino" / "model.xml"
    results_path = PROJECT_ROOT / "results" / "latency_benchmark.json"

    # Check files exist
    print("\nChecking model files...")
    for path, name in [(pytorch_path, "PyTorch"), (onnx_path, "ONNX"), (openvino_path, "OpenVINO")]:
        if path.exists():
            print(f"  ✓ {name}: {path}")
        else:
            print(f"  ✗ {name}: {path} NOT FOUND")
            print(f"\n  Download from Google Drive to: {path}")
            return

    # Create dummy input (batch=1, channels=3, height=224, width=224)
    print("\nCreating test input...")
    input_tensor = torch.randn(1, 3, 224, 224)
    input_array = input_tensor.numpy().astype(np.float32)
    print(f"  Input shape: {input_array.shape}")

    # System info
    import platform
    print(f"\nSystem Info:")
    print(f"  Platform: {platform.system()} {platform.release()}")
    print(f"  Processor: {platform.processor()}")
    print(f"  Python: {platform.python_version()}")
    print(f"  PyTorch: {torch.__version__}")

    num_runs = 1000
    warmup = 100
    print(f"\nBenchmark Settings:")
    print(f"  Warmup runs: {warmup}")
    print(f"  Timed runs: {num_runs}")

    results = {}

    # PyTorch benchmark
    print("\n" + "-" * 40)
    print("Benchmarking PyTorch CPU...")
    from src.model import create_model
    model = create_model(num_classes=6, pretrained=False)
    checkpoint = torch.load(pytorch_path, map_location="cpu")
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    pytorch_results = benchmark_pytorch(model, input_tensor, num_runs, warmup)
    results["pytorch"] = pytorch_results
    print(f"  Median: {pytorch_results['median_ms']:.2f} ms")
    print(f"  Mean:   {pytorch_results['mean_ms']:.2f} ms ± {pytorch_results['std_ms']:.2f}")

    # ONNX Runtime benchmark
    print("\n" + "-" * 40)
    print("Benchmarking ONNX Runtime CPU...")
    try:
        onnx_results = benchmark_onnx(onnx_path, input_array, num_runs, warmup)
        results["onnx"] = onnx_results
        print(f"  Median: {onnx_results['median_ms']:.2f} ms")
        print(f"  Mean:   {onnx_results['mean_ms']:.2f} ms ± {onnx_results['std_ms']:.2f}")
    except ImportError:
        print("  ERROR: onnxruntime not installed. Run: pip install onnxruntime")
        results["onnx"] = None

    # OpenVINO benchmark
    print("\n" + "-" * 40)
    print("Benchmarking OpenVINO CPU...")
    try:
        openvino_results = benchmark_openvino(openvino_path, input_array, num_runs, warmup)
        results["openvino"] = openvino_results
        print(f"  Median: {openvino_results['median_ms']:.2f} ms")
        print(f"  Mean:   {openvino_results['mean_ms']:.2f} ms ± {openvino_results['std_ms']:.2f}")
    except ImportError:
        print("  ERROR: openvino not installed. Run: pip install openvino")
        results["openvino"] = None

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    pytorch_ms = results["pytorch"]["median_ms"]
    print(f"\n{'Runtime':<20} {'Median (ms)':<15} {'Target':<15} {'Status':<10} {'Speedup':<10}")
    print("-" * 70)
    print(f"{'PyTorch':<20} {pytorch_ms:<15.2f} {'(baseline)':<15} {'-':<10} {'1.00x':<10}")

    if results.get("onnx"):
        onnx_ms = results["onnx"]["median_ms"]
        onnx_speedup = pytorch_ms / onnx_ms
        onnx_status = "✓ PASS" if onnx_ms < 8 else "✗ FAIL"
        print(f"{'ONNX Runtime':<20} {onnx_ms:<15.2f} {'<8ms':<15} {onnx_status:<10} {onnx_speedup:.2f}x")
        results["onnx_speedup"] = onnx_speedup

    if results.get("openvino"):
        ov_ms = results["openvino"]["median_ms"]
        ov_speedup = pytorch_ms / ov_ms
        ov_status = "✓ PASS" if ov_ms < 5 else "✗ FAIL"
        print(f"{'OpenVINO':<20} {ov_ms:<15.2f} {'<5ms':<15} {ov_status:<10} {ov_speedup:.2f}x")
        results["openvino_speedup"] = ov_speedup

    # Save results
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # Resume bullet helper
    print("\n" + "=" * 60)
    print("FOR RESUME BULLET B1:")
    print("=" * 60)
    if results.get("onnx") and results.get("openvino"):
        print(f"""
Engineered an OpenCV + EfficientNet-B0 defect classification pipeline
on the NEU Surface Defect dataset achieving 99.54% top-1 accuracy across
6 defect classes; exported to ONNX Runtime ({results['onnx']['median_ms']:.1f}ms) and
OpenVINO IR ({results['openvino']['median_ms']:.1f}ms CPU latency), achieving a
{results['openvino_speedup']:.1f}x speedup over baseline PyTorch.
""")


if __name__ == "__main__":
    main()
