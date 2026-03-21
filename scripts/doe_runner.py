"""
DOE Runner: Run 27-configuration Design of Experiments.

Task 3.3 — Runs all DOE configurations, evaluates classifier on each,
and saves per-class F1 scores to results/doe_results.csv.

Usage (Colab with GPU for faster inference):
    python scripts/doe_runner.py --data_dir /path/to/NEU-DET --model_path /path/to/model.onnx

Or locally:
    python scripts/doe_runner.py --data_dir data/raw/NEU-DET/train/images --model_path models/model.onnx
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.optical_simulator import (
    get_doe_configurations,
    apply_all_degradations,
    DOE_PSF_LEVELS,
    DOE_SNR_LEVELS,
    DOE_VIGNETTE_LEVELS,
)
from src.dataset import NEUDataset, CLASS_NAMES
from src.preprocessing import preprocess_for_tensor


def load_onnx_model(model_path: str):
    """Load ONNX model for inference."""
    import onnxruntime as ort

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    # Use GPU if available
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(str(model_path), sess_options, providers=providers)

    print(f"Loaded ONNX model: {model_path}")
    print(f"Providers: {session.get_providers()}")

    return session


def run_inference(session, images: np.ndarray) -> np.ndarray:
    """Run batch inference on ONNX model."""
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: images.astype(np.float32)})
    return outputs[0]


def compute_f1_scores(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 6):
    """Compute per-class F1 scores."""
    from sklearn.metrics import f1_score

    # Per-class F1
    f1_per_class = f1_score(y_true, y_pred, average=None, labels=range(num_classes))

    # Macro F1
    f1_macro = f1_score(y_true, y_pred, average="macro")

    return f1_per_class, f1_macro


def run_doe(
    data_dir: str,
    model_path: str,
    output_path: str,
    batch_size: int = 32,
):
    """
    Run full DOE experiment.

    Args:
        data_dir: Path to dataset (images folder)
        model_path: Path to ONNX model
        output_path: Path to save results CSV
        batch_size: Batch size for inference
    """
    print("=" * 60)
    print("DOE Runner - 27 Configuration Experiment")
    print("=" * 60)

    # Load model
    session = load_onnx_model(model_path)

    # Load test dataset (without preprocessing - we'll degrade then preprocess)
    print(f"\nLoading dataset from: {data_dir}")
    dataset = NEUDataset(data_dir, split="test", preprocess=False)
    print(f"Test images: {len(dataset)}")

    # Load raw images
    print("Loading raw images...")
    raw_images = []
    labels = []
    for img_path, label in tqdm(dataset.images, desc="Loading"):
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            raw_images.append(img)
            labels.append(label)
    labels = np.array(labels)
    print(f"Loaded {len(raw_images)} images")

    # Get DOE configurations
    configs = get_doe_configurations()
    print(f"\nRunning {len(configs)} DOE configurations...")

    # Results storage
    results = []

    total_start = time.time()

    for config_idx, config in enumerate(configs):
        psf = config["psf_sigma"]
        snr = config["snr_db"]
        vig = config["vignette_strength"]

        print(f"\n[{config_idx + 1}/27] PSF={psf}px, SNR={snr}dB, Vignette={vig}")

        # Apply degradations and preprocess
        processed_images = []
        for img in tqdm(raw_images, desc="Degrading", leave=False):
            # Apply optical degradations
            degraded = apply_all_degradations(
                img,
                psf_sigma=psf,
                snr_db=snr,
                vignette_strength=vig
            )

            # Preprocess for model
            tensor, _ = preprocess_for_tensor(degraded)
            processed_images.append(tensor)

        processed_images = np.stack(processed_images, axis=0)

        # Run inference in batches
        all_preds = []
        for i in range(0, len(processed_images), batch_size):
            batch = processed_images[i:i + batch_size]
            logits = run_inference(session, batch)
            preds = logits.argmax(axis=1)
            all_preds.extend(preds)
        all_preds = np.array(all_preds)

        # Compute F1 scores
        f1_per_class, f1_macro = compute_f1_scores(labels, all_preds)

        print(f"  Macro F1: {f1_macro:.4f}")
        print(f"  Per-class: {', '.join([f'{f:.3f}' for f in f1_per_class])}")

        # Store results (one row per class per config)
        for class_idx, f1 in enumerate(f1_per_class):
            results.append({
                "config_id": config_idx,
                "psf_sigma": psf,
                "snr_db": snr,
                "vignette_strength": vig,
                "class": CLASS_NAMES[class_idx],
                "class_idx": class_idx,
                "f1_score": f1,
                "macro_f1": f1_macro,
            })

    total_time = time.time() - total_start
    print(f"\n\nDOE completed in {total_time / 60:.1f} minutes")

    # Save results
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"Results saved to: {output_path}")
    print(f"Total rows: {len(results)} (27 configs x 6 classes)")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run DOE experiment")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/raw/NEU-DET/train/images",
        help="Path to dataset images folder"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/model.onnx",
        help="Path to ONNX model"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/doe_results.csv",
        help="Output CSV path"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for inference"
    )

    args = parser.parse_args()

    run_doe(
        data_dir=args.data_dir,
        model_path=args.model_path,
        output_path=args.output,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
