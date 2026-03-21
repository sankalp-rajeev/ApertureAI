"""
FastAPI dashboard backend for ApertureAI.

Serves the inspection dashboard and provides API endpoints for:
- Single image prediction with Grad-CAM
- SPC monitoring and demo
- DOE results
- System metrics
"""

import base64
import io
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import onnxruntime as ort
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.spc import SPCMonitor
from src.model import load_model

# Class names (NEU-DET dataset, alphabetical)
CLASS_NAMES = ["Crazing", "Inclusion", "Patches", "Pitted", "Rolled", "Scratches"]

# Initialize FastAPI
app = FastAPI(title="ApertureAI Dashboard", version="1.0.0")

# Serve static files
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Global state
spc_monitor = SPCMonitor(batch_size=25)


@app.on_event("startup")
async def load_models():
    # ONNX — fast inference
    app.state.onnx_sess = ort.InferenceSession(str(PROJECT_ROOT / "models" / "model.onnx"),
                                                providers=["CPUExecutionProvider"])
    
    # PyTorch — for Grad-CAM only (reuse project's load_model which
    # handles checkpoint["model_state_dict"] and 3-channel grayscale arch)
    app.state.pt_model = load_model(
        str(PROJECT_ROOT / "models" / "best_model.pth"),
        num_classes=6,
        device=torch.device("cpu"),
    )


def preprocess_for_inference(img_bytes: bytes) -> tuple:
    """
    Read image bytes and prepare for ONNX inference.

    Returns:
        (input_tensor, display_image) where input_tensor is (1,3,224,224)
        float32 and display_image is the original grayscale for visualization.
    """
    # Decode image
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Could not decode image")

    display_img = img.copy()

    # Apply full training preprocessing pipeline
    from src.preprocessing import preprocess_for_tensor
    tensor_np, _ = preprocess_for_tensor(img, target_size=(224, 224))
    
    # Add batch dimension: (1, 3, 224, 224)
    input_tensor = tensor_np[np.newaxis, ...]
    
    return input_tensor, display_img


def compute_gradcam(model, img_tensor: torch.Tensor) -> np.ndarray:
    """img_tensor: (1, 1, 224, 224) float32"""
    gradients = []
    activations = []

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    def forward_hook(module, input, output):
        activations.append(output)

    # Hook into last conv block of EfficientNet-B0
    target_layer = model.features[8]
    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_full_backward_hook(backward_hook)

    img_tensor.requires_grad_(True)
    output = model(img_tensor)
    pred_class = output.argmax(dim=1).item()
    
    model.zero_grad()
    output[0, pred_class].backward()

    fh.remove()
    bh.remove()

    # Weighted combination of feature maps
    grads = gradients[0].detach()          # (1, C, H, W)
    acts  = activations[0].detach()        # (1, C, H, W)
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = (weights * acts).sum(dim=1, keepdim=False)
    cam = torch.relu(cam).squeeze().numpy()

    # Normalize and resize to 224x224
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    cam = cv2.resize(cam, (224, 224))
    cam = (cam * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    return heatmap  # BGR uint8


# ============================================================
# Routes
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the dashboard."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return HTMLResponse("<h1>ApertureAI Dashboard</h1><p>Static files not found.</p>")


@app.post("/api/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    """
    Single image prediction with saliency heatmap.

    Returns class, confidence, per-class probabilities, and heatmap overlay.
    """
    try:
        contents = await file.read()
        input_tensor, display_img = preprocess_for_inference(contents)

        session = request.app.state.onnx_sess
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: input_tensor})
        logits = outputs[0][0]

        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probabilities = exp_logits / exp_logits.sum()

        pred_class = int(np.argmax(probabilities))
        confidence = float(probabilities[pred_class])

        # Generate Grad-CAM heatmap
        pt_img_tensor = torch.from_numpy(input_tensor).float()
        heatmap_bgr = compute_gradcam(request.app.state.pt_model, pt_img_tensor)

        # Blend with original for overlay
        original_rgb = cv2.resize(display_img, (224, 224))
        if len(original_rgb.shape) == 2:
            original_rgb = cv2.cvtColor(original_rgb, cv2.COLOR_GRAY2BGR)
        overlay = cv2.addWeighted(original_rgb, 0.5, heatmap_bgr, 0.5, 0)

        # Encode to base64
        _, buf = cv2.imencode(".png", overlay)
        gradcam_b64 = base64.b64encode(buf).decode("utf-8")

        # Encode original image
        _, orig_buffer = cv2.imencode(".png", original_rgb)
        orig_b64 = base64.b64encode(orig_buffer).decode("utf-8")

        return {
            "class_name": CLASS_NAMES[pred_class],
            "class_idx": pred_class,
            "confidence": round(confidence, 4),
            "probabilities": {name: round(float(p), 4)
                              for name, p in zip(CLASS_NAMES, probabilities)},
            "heatmap_base64": gradcam_b64,
            "original_base64": orig_b64,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/spc/run-demo")
async def spc_demo():
    """
    Run a 40-batch SPC demo with drift injection.

    Pre-computes all batches server-side and returns the full series.
    Client animates with setInterval.
    """
    np.random.seed(42)
    monitor = SPCMonitor(batch_size=25)

    baseline_rate = 0.08
    drift1_rate = 0.18
    drift2_rate = 0.28

    batches = []

    # Phase A: 20 baseline batches
    for i in range(20):
        batch = (np.random.rand(25) < baseline_rate).astype(int).tolist()
        result = monitor.add_batch(batch)
        batches.append({
            "batch_idx": i,
            "phase": "baseline",
            "xbar": result.xbar,
            "r_value": result.r_value,
            "ooc": result.ooc,
            "rules": result.rules_triggered,
        })

    # Compute control limits after baseline
    limits = monitor.compute_control_limits()

    # Phase B: 10 batches with +10% drift
    for i in range(10):
        batch = (np.random.rand(25) < drift1_rate).astype(int).tolist()
        result = monitor.add_batch(batch)
        batches.append({
            "batch_idx": 20 + i,
            "phase": "drift_10pct",
            "xbar": result.xbar,
            "r_value": result.r_value,
            "ooc": result.ooc,
            "rules": result.rules_triggered,
        })

    # Phase C: 10 batches with +20% drift
    for i in range(10):
        batch = (np.random.rand(25) < drift2_rate).astype(int).tolist()
        result = monitor.add_batch(batch)
        batches.append({
            "batch_idx": 30 + i,
            "phase": "drift_20pct",
            "xbar": result.xbar,
            "r_value": result.r_value,
            "ooc": result.ooc,
            "rules": result.rules_triggered,
        })

    # Compute alert summary
    drift_batches = [b for b in batches if b["phase"] != "baseline"]
    detections = sum(1 for b in drift_batches if b["ooc"])

    return {
        "batches": batches,
        "limits": {
            "cl_x": limits["cl_x"],
            "ucl_x": limits["ucl_x"],
            "lcl_x": limits["lcl_x"],
            "ucl_r": limits["ucl_r"],
        },
        "summary": {
            "total_batches": len(batches),
            "baseline_batches": 20,
            "drift_batches": 20,
            "detections": detections,
            "detection_rate": detections / 20 if drift_batches else 0,
        },
    }


@app.get("/api/doe/results")
async def doe_results():
    """Return DOE results for dashboard plotting."""
    csv_path = PROJECT_ROOT / "results" / "doe_results.csv"
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="DOE results not found. Run doe_runner.py first.")

    import pandas as pd
    df = pd.read_csv(csv_path)

    # Main effects
    psf_effects = df.groupby("psf_sigma")["macro_f1"].mean().to_dict()
    snr_effects = df.groupby("snr_db")["macro_f1"].mean().to_dict()
    vig_effects = df.groupby("vignette_strength")["macro_f1"].mean().to_dict()

    # Interaction heatmap data (PSF × SNR)
    interaction = df.groupby(["psf_sigma", "snr_db"])["macro_f1"].mean().reset_index()

    # Class sensitivity
    class_data = df.groupby(["psf_sigma", "class"])["f1_score"].mean().reset_index()

    return {
        "main_effects": {
            "psf": {str(k): round(v, 4) for k, v in psf_effects.items()},
            "snr": {str(k): round(v, 4) for k, v in snr_effects.items()},
            "vignette": {str(k): round(v, 4) for k, v in vig_effects.items()},
        },
        "interaction": interaction.to_dict(orient="records"),
        "class_sensitivity": class_data.to_dict(orient="records"),
        "summary": {
            "best_f1": round(float(df["macro_f1"].max()), 4),
            "worst_f1": round(float(df["macro_f1"].min()), 4),
            "total_configs": int(df["config_id"].nunique()),
            "psf_threshold": 3,  # from DOE analysis
        },
    }


@app.get("/api/metrics")
async def metrics():
    """Return latency benchmarks and model info."""
    latency_path = PROJECT_ROOT / "results" / "latency_benchmark.json"
    if latency_path.exists():
        with open(latency_path) as f:
            latency = json.load(f)
    else:
        latency = {}

    return {
        "latency": latency,
        "model": {
            "architecture": "EfficientNet-B0",
            "input_size": "224×224",
            "classes": CLASS_NAMES,
            "num_classes": len(CLASS_NAMES),
            "accuracy": 0.9954,
            "dataset": "NEU-DET (1,200 training, 600 test)",
        },
        "doe_summary": {
            "resolution_threshold_px": 3,
            "f1_degradation_pct": 50.0,
            "dominant_factor": "PSF blur",
            "most_sensitive_class": "Crazing",
        },
    }


if __name__ == "__main__":
    import uvicorn
    print("Starting ApertureAI Dashboard at http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
