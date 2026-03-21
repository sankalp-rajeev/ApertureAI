# Project Context & Progress Tracker

## Overview
**Project:** PCB Surface Defect Inspection & SPC Monitoring System
**Goal:** 3 resume bullets targeting Intel machine vision roles
**Estimated Time:** 12-16 focused hours

---

## Phase 1 — Data & Preprocessing (3-4 hrs)

### Task 1.1 — Dataset Setup
- [ ] Download NEU-DET dataset
- [ ] Write `dataset_audit.py` to check class balance, image sizes, corrupted files
- [ ] Run audit and verify: 0 errors, 1,800 images confirmed

### Task 1.2 — OpenCV Preprocessing Pipeline
- [ ] Implement `preprocess(img)` in `src/preprocessing.py`
- [ ] CLAHE: `clip_limit=2.0`, `tile_grid=(8,8)`
- [ ] Adaptive threshold: `block_size=11`, `C=2`
- [ ] Morphological ops: closing kernel (5×5) → opening kernel (3×3)
- [ ] Contour extraction with `RETR_EXTERNAL`, `CHAIN_APPROX_SIMPLE`
- [ ] ROI logic: largest contour bbox + 10px padding, fallback to center crop
- [ ] Visual QA pass + ROI hit rate >90% on 180-image sample

### Task 1.3 — PyTorch Dataset Class
- [ ] Implement `NEUDataset(root, split, transform)` in `src/dataset.py`
- [ ] Stratified 70/15/15 split
- [ ] Verify DataLoader produces `(B, 3, 224, 224)` tensors
- [ ] One full batch loads without error

### Task 1.4 — Augmentation Pipeline
- [ ] albumentations Compose with rotation, flip, brightness, blur, elastic
- [ ] Visualize augmented samples
- [ ] Augmented images look physically plausible

---

## Phase 2 — Classifier Training (3-4 hrs)

### Task 2.1 — Model Setup
- [ ] Load EfficientNet-B0 with ImageNet weights
- [ ] Grayscale adaptation: `weight = pretrained_weight.mean(dim=1, keepdim=True)`
- [ ] Replace final fc: `nn.Linear(1280, 6)`
- [ ] Forward pass on dummy batch completes without shape error

### Task 2.2 — Training Loop
- [ ] Mixed precision with `torch.cuda.amp.autocast` + `GradScaler`
- [ ] AdamW optimizer with weight decay 1e-4
- [ ] Cosine annealing LR: 1e-4 → 1e-3 → 1e-5 over 50 epochs
- [ ] Label smoothing (ε=0.1)
- [ ] Early stopping patience=10 on validation F1
- [ ] Log loss, accuracy, per-class F1 each epoch → CSV
- [ ] 50 epochs without NaN loss

### Task 2.3 — Hyperparameter Tuning
- [ ] Experiment 1: LR schedule variations
- [ ] Experiment 2: Label smoothing values
- [ ] Experiment 3: Augmentation intensity
- [ ] Val F1 >96% on best config

### Task 2.4 — Evaluation
- [ ] Confusion matrix on test set
- [ ] Per-class precision/recall/F1
- [ ] CPU latency benchmark: `timeit` over 1,000 single-image passes
- [ ] **Exit:** Test accuracy ≥97%, CPU latency <8ms

### Task 2.5 — Grad-CAM
- [ ] Hook into last conv layer (`features.8`)
- [ ] Implement gradient computation and weighted feature maps
- [ ] Generate heatmap overlays for 5 samples/class (30 total)
- [ ] >80% heatmaps qualitatively highlight defect region

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
- [ ] For each config: degrade test set → batch inference → record per-class F1
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
- [ ] Panel 5: Yield dashboard with CSV export
- [ ] All panels render without error

### Task 4.5 — Integration Test
- [ ] Full pipeline end-to-end
- [ ] Measure total latency
- [ ] **Exit:** <2 seconds per image, no uncaught exceptions

---

## Final Metrics (Fill After Completion)

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Top-1 Accuracy | ≥97% | | |
| CPU Inference Latency | <8ms | | |
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
| | | |

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
> Engineered an **OpenCV + EfficientNet-B0** defect classification pipeline on the NEU Surface Defect dataset achieving **[X]% top-1 accuracy** across 6 defect classes with **[Y]ms CPU inference latency** per frame, using CLAHE normalization, adaptive thresholding, and contour-guided ROI extraction.

**B2:**
> Designed an **optical degradation simulator** modeling PSF convolution (Airy disk, NA 0.1–0.5), Poisson + Gaussian sensor noise, and illumination vignetting; ran a **27-configuration DOE** (3-factor × 3-level), identifying PSF σ=[Z]px as the resolution threshold beyond which F1 degrades by **[W]%**, informing camera/lens selection trade-offs for high-volume inspection.

**B3:**
> Implemented **Statistical Process Control** (X-bar/R-charts, 4 Western Electric rules) over simulated production batches achieving **zero false escapes** across 500 images; deployed SSIM-based frame pre-filter reducing classifier load by **[V]%** with zero missed defects; surfaced Grad-CAM heatmaps, OOC alerts, and batch yield in a **Streamlit dashboard**.
