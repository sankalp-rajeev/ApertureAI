# Project Context & Progress Tracker

## Overview
**Project:** PCB Surface Defect Inspection & SPC Monitoring System (PRD v1.1)
**Goal:** 3 resume bullets targeting Intel machine vision roles
**Estimated Time:** 12-16 focused hours

---

## Phase 1 — Data & Preprocessing (3-4 hrs) ✅ COMPLETE

### Task 1.1 — Dataset Setup
- [x] Download NEU-DET dataset
- [x] Write `dataset_audit.py` to check class balance, image sizes, corrupted files
- [x] Run audit and verify: 0 errors, 1,800 images confirmed

### Task 1.2 — OpenCV Preprocessing Pipeline
- [x] Implement `preprocess(img)` in `src/preprocessing.py`
- [x] CLAHE: `clip_limit=2.0`, `tile_grid=(8,8)`
- [x] Adaptive threshold: `block_size=11`, `C=2`
- [x] Morphological ops: closing kernel (5×5) → opening kernel (3×3)
- [x] Contour extraction with `RETR_EXTERNAL`, `CHAIN_APPROX_SIMPLE`
- [x] ROI logic: largest contour bbox + 10px padding, fallback to center crop
- [x] Visual QA pass + ROI hit rate >90% on 180-image sample

### Task 1.3 — PyTorch Dataset Class
- [x] Implement `NEUDataset(root, split, transform)` in `src/dataset.py`
- [x] Stratified 70/15/15 split
- [x] Verify DataLoader produces `(B, 3, 224, 224)` tensors
- [x] One full batch loads without error

### Task 1.4 — Augmentation Pipeline
- [x] albumentations Compose with rotation, flip, brightness, blur, elastic
- [x] Visualize augmented samples
- [x] Augmented images look physically plausible

---

## Phase 2 — Classifier Training & Inference Optimization (4-5 hrs) ✅ COMPLETE

### Task 2.1 — Model Setup
- [x] Load EfficientNet-B0 with ImageNet weights
- [x] Grayscale adaptation: `weight = pretrained_weight.mean(dim=1, keepdim=True)`
- [x] Replace final fc: `nn.Linear(1280, 6)`
- [x] Forward pass on dummy batch completes without shape error

### Task 2.2 — Training Loop
- [x] Mixed precision with `torch.cuda.amp.autocast` + `GradScaler`
- [x] AdamW optimizer with weight decay 1e-4
- [x] Cosine annealing LR: 1e-4 → 1e-3 → 1e-5 over 50 epochs
- [x] Label smoothing (ε=0.1)
- [x] Early stopping patience=10 on validation F1
- [x] Log loss, accuracy, per-class F1 each epoch → CSV
- [x] 50 epochs without NaN loss (early stopped at epoch 13)

### Task 2.3 — Hyperparameter Tuning
- [x] Default config achieved Val F1 = 1.0000, no tuning needed

### Task 2.4 — Evaluation
- [x] Confusion matrix on test set
- [x] Per-class precision/recall/F1
- [x] PyTorch CPU baseline latency: 33.21ms (1,000 runs)
- [x] **Exit:** Test accuracy 99.54% ✅

### Task 2.5 — Grad-CAM
- [x] Hook into last conv layer (`features.8`)
- [x] Implement gradient computation and weighted feature maps
- [x] Generate heatmap overlays for 5 samples/class (30 total)
- [x] >80% heatmaps qualitatively highlight defect region

### Task 2.6 — ONNX Export & Validation
- [x] Export with `torch.onnx.export`, opset 17, dynamic batch axis
- [x] input_names=["image"], output_names=["logits"]
- [x] Validate: run ONNX Runtime on test set
- [x] Compare class labels to PyTorch — predictions match
- [x] Max abs logit diff: 0.001 (acceptable)
- [x] Benchmark ONNX Runtime latency: 30.37ms
- [x] **Exit:** ONNX working (latency above 8ms target due to hardware)

### Task 2.7 — OpenVINO Conversion & Benchmark
- [x] Convert: `ov.save_model()` → .xml + .bin
- [x] Compile for CPU: `core.compile_model(model, "CPU")`
- [x] Benchmark OpenVINO Runtime latency: 12.05ms
- [x] Save three-way comparison to `results/latency_benchmark.json`
- [x] **Exit:** OpenVINO speedup 2.76x over PyTorch ✅

---

## Phase 3 — Optical Simulator & DOE (3-4 hrs)

### Task 3.1 — Degradation Functions
- [ ] Implement `apply_psf(img, sigma)` — Gaussian blur
- [ ] Implement `apply_noise(img, snr_db)` — Poisson + Gaussian
- [ ] Implement `apply_vignette(img, strength)` — radial falloff
- [ ] Implement `apply_dof_blur(img, z_offset)` — defocus
- [ ] Unit tests: `apply_psf(img, sigma=0)` returns identical image
- [ ] All 4 functions pass unit tests

### Task 3.2 — PSF Physics Validation
- [ ] Verify: σ_pixels = 0.61·λ / (NA·pixel_size)
- [ ] For pixel_size=5μm, λ=550nm:
  - [ ] NA=0.5 → σ≈1.3px
  - [ ] NA=0.2 → σ≈3.4px
  - [ ] NA=0.1 → σ≈6.7px
- [ ] Document derivation in README
- [ ] Computed σ values match theory within 10%

### Task 3.3 — DOE Runner
- [ ] Generate 27 combinations with `itertools.product`
- [ ] DOE factors:
  - PSF sigma: [1, 3, 6] px
  - SNR: [40, 30, 20] dB
  - Vignette: [0, 0.10, 0.25]
- [ ] For each config: degrade test set → ONNX inference → record per-class F1
- [ ] Save to `results/doe_results.csv`
- [ ] CSV has 27×6=162 rows, no NaN values

### Task 3.4 — DOE Analysis
- [ ] Main effects plots: 3 subplots (factor level vs mean F1)
- [ ] Interaction heatmap: PSF × SNR with F1 as color
- [ ] `statsmodels` OLS for factor significance
- [ ] Identify dominant factor with p<0.05
- [ ] Find PSF σ threshold where F1 < 0.90

---

## Phase 4 — SPC & Dashboard (3-4 hrs)

### Task 4.1 — SPC Implementation
- [ ] Implement `SPCMonitor` class in `src/spc.py`
- [ ] X-bar chart: UCL/LCL = X̄ ± 3σ
- [ ] R chart: range tracking within batch
- [ ] Western Electric Rule 1: 1 point beyond 3σ
- [ ] Western Electric Rule 2: 9 consecutive same side of centerline
- [ ] Western Electric Rule 3: 6 consecutive monotonic
- [ ] Western Electric Rule 4: 2 of 3 beyond 2σ
- [ ] Correct UCL/LCL on synthetic data with known σ

### Task 4.2 — Drift Injection Validation
- [ ] Batches 1-20: baseline from clean test set
- [ ] Batches 21-30: inject +5% defects
- [ ] Batches 31-40: inject +15% defects + class shift
- [ ] **Exit:** False escape rate = 0%, false alarm rate <5%

### Task 4.3 — SSIM Pre-filter
- [ ] Compute SSIM vs defect-free reference template
- [ ] Find threshold for 0 missed defects
- [ ] Measure % frames skipped
- [ ] **Exit:** ≥80% frames skipped, 0 missed defects

### Task 4.4 — Streamlit Dashboard
- [ ] Panel 1: Image upload + prediction + Grad-CAM heatmap
- [ ] Panel 2: Batch inspection with per-class counts
- [ ] Panel 3: SPC X-bar/R-charts with OOC alerts
- [ ] Panel 4: DOE main effects + interaction heatmap
- [ ] Panel 5: Yield dashboard + latency comparison + CSV export
- [ ] All panels render without error

### Task 4.5 — Integration Test
- [ ] Full pipeline end-to-end
- [ ] Measure total latency
- [ ] **Exit:** <2 seconds per image, no uncaught exceptions

---

## Final Metrics (Fill After Completion)

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Top-1 Accuracy | ≥97% | **99.54%** | ✅ PASS |
| PyTorch CPU Latency (baseline) | Measure | 33.21ms | ✅ Measured |
| ONNX Runtime CPU Latency | <8ms | 30.37ms | ⚠️ Above target |
| OpenVINO Runtime CPU Latency | <5ms | 12.05ms | ⚠️ Above target |
| PyTorch → OpenVINO Speedup | ≥2x | **2.76x** | ✅ PASS |
| DOE Configurations | 27 | | |
| Resolution Threshold (PSF σ) | Report | | |
| F1 Degradation (worst config) | Report % | | |
| SPC False Escape Rate | 0% | | |
| Manual Review Reduction | ≥80% | | |

---

## Errors & Issues Log

Track any errors, bugs, or issues encountered for future learning.

### Phase 1

| Date | Issue | Resolution |
|------|-------|------------|
| | | |

### Phase 2

| Date | Issue | Resolution |
|------|-------|------------|
| 2026-03-20 | Only 2/6 class folders uploaded to Drive | Uploaded remaining 4 folders |
| 2026-03-20 | Dataset finding files with class subfolders | Updated `find_all_images()` with `CLASS_FOLDER_MAP` |
| 2026-03-20 | GaussNoise `var_limit` deprecated in albumentations | Changed to `std_range` |
| 2026-03-20 | ONNX export needed `onnxscript` module | `pip install onnxscript` |
| 2026-03-20 | OpenVINO `mo` import changed | Use `import openvino as ov` and `ov.save_model()` |
| 2026-03-20 | ONNX model.onnx.data file missing locally | Download external weights file from Drive |

### Phase 3

| Date | Issue | Resolution |
|------|-------|------------|
| | | |

### Phase 4

| Date | Issue | Resolution |
|------|-------|------------|
| | | |

---

## Notes & Learnings

*Add any insights, tips, or learnings as you progress through the project.*

---

## Resume Bullets (Final)

Fill in after completing all phases:

**B1:**
> Engineered an **OpenCV + EfficientNet-B0** defect classification pipeline on the NEU Surface Defect dataset achieving **99.54% top-1 accuracy** across 6 defect classes; exported to **ONNX Runtime** (30.4ms) and **OpenVINO IR** (12.0ms CPU latency), achieving a **2.8x speedup** over baseline PyTorch — using CLAHE normalization, adaptive thresholding, and contour-guided ROI extraction.

**B2:**
> Designed an **optical degradation simulator** modeling PSF convolution (Airy disk, NA 0.1–0.5), Poisson + Gaussian sensor noise, and illumination vignetting; ran a **27-configuration DOE** (3-factor × 3-level), identifying PSF σ=[A]px as the resolution threshold beyond which F1 degrades by **[B]%**, informing camera/lens selection trade-offs for high-volume inspection.

**B3:**
> Implemented **Statistical Process Control** (X-bar/R-charts, 4 Western Electric rules) over simulated production batches achieving **zero false escapes** across 500 images; deployed SSIM-based frame pre-filter reducing classifier load by **[V]%** with zero missed defects; surfaced Grad-CAM heatmaps, OOC alerts, and batch yield in a **Streamlit dashboard**.
