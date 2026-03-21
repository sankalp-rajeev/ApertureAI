# Product Requirements Document
## PCB Surface Defect Inspection & SPC Monitoring System

| Field | Value |
|---|---|
| **Author** | Sankalp Rajeev |
| **Version** | 1.0 |
| **Date** | March 2026 |
| **Status** | APPROVED FOR BUILD |

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement & Motivation](#2-problem-statement--motivation)
3. [Goals & Non-Goals](#3-goals--non-goals)
4. [Target Resume Bullets — The North Star](#4-target-resume-bullets--the-north-star)
5. [Dataset](#5-dataset)
6. [System Architecture](#6-system-architecture)
7. [Detailed Module Specifications](#7-detailed-module-specifications)
8. [Tech Stack & Engineering Decisions](#8-tech-stack--engineering-decisions)
9. [Developer Tooling — GitHub Copilot & Google Colab Pro](#9-developer-tooling--github-copilot--google-colab-pro)
10. [Build Phases & Detailed Task Breakdown](#10-build-phases--detailed-task-breakdown)
11. [Quantification Summary](#11-quantification-summary)
12. [Risks & Mitigations](#12-risks--mitigations)
13. [Final Resume Bullets — Post-Build](#13-final-resume-bullets--post-build)

---

## 1. Executive Summary

This project builds an end-to-end automated visual inspection system for manufacturing defect detection and classification, directly mirroring the engineering challenges faced in Intel's end-of-line singulated package inspection workflows. The system combines a classical OpenCV preprocessing pipeline, a fine-tuned CNN classifier (EfficientNet-B0), an optical degradation simulator grounded in imaging physics, and a Statistical Process Control (SPC) monitoring layer — all surfaced through a Streamlit dashboard.

> **Scope constraint:** The project is designed to produce exactly **three resume bullets** that collectively cover six of Intel's stated preferred qualifications: machine vision, OpenCV, CNN classification, SPC/process control, data visualization, and DOE-based optics analysis. Everything outside those three bullets is out of scope.

---

## 2. Problem Statement & Motivation

Manual visual inspection of manufactured surfaces is slow, inconsistent, and non-scalable. Automated machine vision systems must handle six core engineering challenges that this project directly models:

| Challenge | Real-World Manifestation | What We Model | Intel JD Mapping |
|---|---|---|---|
| Defect variability | 6 morphologically distinct defect classes | NEU 6-class dataset, 1,800 images | Defect detection & classification |
| Optical degradation | Lens aberration, depth-of-field blur, sensor noise | PSF convolution, Poisson + Gaussian noise | Optics fundamentals, light-matter interaction |
| Illumination non-uniformity | Directional lighting in dark-field/bright-field setups | Vignetting + gradient illumination simulation | Light source qualification |
| Process drift | Defect rate changes over production batches | SPC X-bar/R-charts over synthetic batches | SPC & process control |
| Parameter sensitivity | Which imaging setting degrades detection most? | 3-factor DOE across PSF, SNR, illumination | DOE & experimental design |
| Inspection throughput | Sub-10ms per frame at line speed | Inference latency benchmarking | High-volume manufacturing |

---

## 3. Goals & Non-Goals

### 3.1 Goals (Resume-Mapped)

Every goal maps to a specific resume bullet. If it doesn't appear in a bullet, it's not a goal.

| Goal | Resume Bullet |
|---|---|
| ≥97% top-1 accuracy on NEU 6-class defect dataset | B1 — CNN classifier quantifier |
| <8ms inference latency per frame (CPU) | B1 — production-grade latency claim |
| Optical DOE: measure F1 degradation across 27 configurations (3 factors × 3 levels) | B2 — DOE quantifier |
| Quantify PSF resolution threshold at which classifier F1 drops below 90% | B2 — optics/imaging claim |
| SPC system: zero false escapes across 500-image test run | B3 — SPC quantifier |
| ≥80% reduction in manual review frames via SSIM pre-filter | B3 — efficiency quantifier |
| Streamlit dashboard with Grad-CAM heatmaps, SPC charts, batch yield tracking | B3 — visualization claim |

### 3.2 Non-Goals

- Real semiconductor/wafer imagery — NEU Surface Defect dataset is the manufacturing proxy
- 3D inspection, profilometry, or structured light — 2D imaging only
- Hardware integration — pure software simulation throughout
- Multi-GPU training — single Colab Pro A100/T4 session is sufficient
- Cloud deployment beyond localhost Streamlit — no production infra needed for resume
- Unit test suite, CI/CD pipeline — not resume-relevant for this project scope

---

## 4. Target Resume Bullets — The North Star

> Everything in this PRD is reverse-engineered from these three bullets. Build only what makes them true and measurable.

**B1:**
> Engineered an **OpenCV + EfficientNet-B0** defect classification pipeline on the NEU Surface Defect dataset achieving **≥97% top-1 accuracy** across 6 defect classes with **<8ms CPU inference latency** per frame, using CLAHE normalization, adaptive thresholding, and contour-guided ROI extraction.

**B2:**
> Designed an **optical degradation simulator** modeling PSF convolution (Airy disk, NA 0.1–0.5), Poisson + Gaussian sensor noise, and illumination vignetting; ran a **27-configuration DOE** (3-factor × 3-level), identifying PSF σ as the resolution threshold beyond which F1 degrades by **[measured %]**, informing camera/lens selection trade-offs for high-volume inspection.

**B3:**
> Implemented **Statistical Process Control** (X-bar/R-charts, 4 Western Electric rules) over simulated production batches achieving **zero false escapes** across 500 images; deployed SSIM-based frame pre-filter reducing classifier load by **≥80%** with zero missed defects; surfaced Grad-CAM heatmaps, OOC alerts, and batch yield in a **Streamlit dashboard**.

---

## 5. Dataset

### 5.1 NEU Surface Defect Database

| Property | Value |
|---|---|
| Source | Northeastern University — publicly available, 500+ academic citations |
| Total images | 1,800 grayscale images (300 per class) |
| Image size | 200×200 pixels |
| Defect classes | Crazing, Inclusion, Patches, Pitted Surface, Rolled-in Scale, Scratches |
| Format | JPEG + PASCAL VOC bounding box annotations |
| Why this dataset | Real industrial steel surface defects; analogous to package inspection defects (scratches ↔ surface marks, inclusions ↔ embedded particles) |

### 5.2 Data Split Strategy

| Split | Size & Rationale |
|---|---|
| Train | 70% (1,260 images) — stratified by class |
| Validation | 15% (270 images) — hyperparameter tuning only |
| Test | 15% (270 images) — held out, never seen during development |
| SPC batch simulation | 500-image synthetic run drawn from test set with injected drift |

### 5.3 Augmentation Strategy

Augmentation must mimic real inspection variability — not introduce unrealistic transforms.

- ✅ Random rotation ±15° — part misalignment on conveyor
- ✅ Horizontal/vertical flip — symmetric defect morphology
- ✅ Random brightness ±20% — illumination variation between batches
- ✅ Gaussian blur (σ=0.5–1.5) — slight focus variation
- ✅ Elastic deformation — surface texture variation
- ❌ Random cropping that removes defect region
- ❌ Color jitter — these are grayscale images

---

## 6. System Architecture

### 6.1 Module Overview

Four independent modules with clean interfaces. Each developed and tested in isolation.

| Module | Input | Output | Maps To |
|---|---|---|---|
| M1: OpenCV Preprocessor | Raw defect image (200×200) | Cleaned ROI tensor | Bullet 1 |
| M2: EfficientNet Classifier | Preprocessed tensor | Class label + confidence + Grad-CAM | Bullet 1 |
| M3: Optical Degradation Simulator | Clean image | Degraded image at specified params | Bullet 2 |
| M4: SPC Monitor | Batch defect counts (time series) | Control charts + OOC alerts | Bullet 3 |
| M5: Streamlit Dashboard | All module outputs | Interactive UI | Bullet 3 |

### 6.2 Data Flow

```
RAW IMAGE
    ↓  [M1] OpenCV Preprocessor
         adaptive_threshold() → morphological_ops() → contour_extract() → ROI crop
    ↓  [M2] EfficientNet-B0 Classifier
         forward_pass() → softmax() → grad_cam() → {label, conf, heatmap}
    ↓  [M4] SPC Monitor  (aggregated per batch of 25 frames)
         xbar_chart() → r_chart() → western_electric_rules() → {OOC flag}
    ↓  [M5] Streamlit Dashboard
         render heatmaps + control charts + yield tracker + CSV export

PARALLEL PATH:
[M3] Optical Simulator → degraded images → M1 → M2  (DOE evaluation loop)
```

---

## 7. Detailed Module Specifications

### 7.1 Module 1 — OpenCV Preprocessing Pipeline

**Purpose:** Isolate defect ROI from background texture, normalize contrast, and produce a clean tensor for the classifier. This is what Intel's inspection recipe engineers do manually — automating it is the resume-relevant claim.

#### Engineering Decisions

| Decision | Rationale & Tradeoff |
|---|---|
| Adaptive thresholding (Gaussian) over global Otsu | NEU images have non-uniform illumination; global threshold fails on dark-edge images. Tradeoff: slightly slower, requires block_size tuning |
| CLAHE over global histogram equalization | Limits noise amplification in flat background regions. Global HE over-enhances background texture, creating false edges |
| Morphological closing before contour detection | Fills small gaps in defect boundaries from thresholding. Opening would erode thin scratch defects |
| Contour-guided ROI crop vs fixed center crop | Defects are not always centered. Fixed crop loses 15–30% of off-center defects in the NEU dataset |
| Grayscale pipeline throughout | NEU images are grayscale. Adding channels adds no signal and triples memory |

#### Implementation Tasks

**Task 1.1 — Dataset audit**
- Load NEU dataset, verify 1,800 images, 6 classes, 300 each
- Write `dataset_audit.py` to check for corrupted files, class balance, image size consistency
- Exit criterion: audit report shows 0 errors

**Task 1.2 — Preprocessing pipeline**
- Implement `preprocess(img) → (roi_tensor, contour_found)`
- CLAHE: `clip_limit=2.0`, `tile_grid=(8,8)` — benchmark vs global HE on 50-image sample
- Adaptive threshold: `block_size=11`, `C=2` — tune on 20 images per class
- Morphological ops: closing kernel (5×5) → opening kernel (3×3) for noise removal
- Contour: `cv2.findContours` with `RETR_EXTERNAL`, `CHAIN_APPROX_SIMPLE`
- ROI logic: largest contour bounding box + 10px padding, fallback to center crop if no contour
- Exit criterion: visual QA pass + ROI hit rate >90% on 180-image sample (IoU > 0.5 with GT box)

**Task 1.3 — PyTorch Dataset class**
- `NEUDataset(root, split, transform)` with stratified 70/15/15 split
- Verify DataLoader produces correct shapes `(B, 3, 224, 224)`
- Exit criterion: one full batch loads without error

---

### 7.2 Module 2 — EfficientNet-B0 Classifier

#### Model Selection Rationale

| Candidate | Decision |
|---|---|
| ResNet-50 | ❌ 25M params — overkill for 200×200 grayscale. Slower CPU inference |
| VGG16 | ❌ 23M params, no skip connections — known to overfit on small industrial datasets |
| **EfficientNet-B0** | ✅ 5.3M params, compound scaling, SoTA on small datasets. <8ms CPU inference feasible |
| MobileNetV3 | ❌ Faster but lower accuracy ceiling. B0 gives better accuracy/latency tradeoff |
| Custom CNN | ❌ No transfer learning benefit. NEU is small (1,800 imgs); pretrained features generalize better |

#### Training Configuration

| Hyperparameter | Value & Rationale |
|---|---|
| Pretrained weights | ImageNet — adapt first conv to grayscale via channel mean replication trick |
| Learning rate schedule | Cosine annealing with warmup: 1e-4 → 1e-3 → 1e-5 over 50 epochs |
| Batch size | 32 — fits in 8GB GPU; larger batch hurts generalization on small datasets |
| Loss function | Cross-entropy with label smoothing (ε=0.1) — prevents overconfidence on 6-class problem |
| Optimizer | AdamW with weight decay 1e-4 — better generalization than Adam on small datasets |
| Early stopping | Patience=10 on validation F1 — prevents overfitting at epoch 30–40 |

#### Grad-CAM Implementation

Grad-CAM is not optional — it makes the project inspection-relevant by showing which region drove classification, directly analogous to defect localization in real tools.

- Hook into last convolutional layer (`features.8`) of EfficientNet-B0
- Compute gradient of target class score w.r.t. feature map activations
- Weight feature maps by global average pooled gradients → ReLU → upsample to 200×200
- Overlay on original image as heatmap — export as PNG for dashboard

#### Quantification Targets

| Metric | Target |
|---|---|
| Top-1 accuracy (test set) | ≥97% |
| Per-class F1 (worst class) | ≥94% |
| CPU inference latency (single image, batch=1) | <8ms |
| GPU inference throughput | >200 fps (batch=32, T4/A100) |
| Model size | <25MB |

#### Implementation Tasks

**Task 2.1 — Model setup**
- Load EfficientNet-B0 with ImageNet weights
- Grayscale → 3-channel: `weight = pretrained_weight.mean(dim=1, keepdim=True)`
- Replace final fc: `nn.Linear(1280, 6)`
- Exit criterion: forward pass on dummy batch completes without shape error

**Task 2.2 — Training loop**
- Mixed precision: `torch.cuda.amp.autocast` + `GradScaler`
- Log loss, accuracy, per-class F1 (sklearn) each epoch → CSV
- Exit criterion: 50 epochs without NaN loss

**Task 2.3 — Evaluation**
- Confusion matrix + per-class precision/recall/F1 on test set
- CPU latency: `timeit` over 1,000 single-image forward passes — report median
- Exit criterion: test accuracy ≥97%, CPU latency <8ms

**Task 2.4 — Grad-CAM**
- Generate heatmap overlays for 5 samples/class (30 total)
- Exit criterion: >80% of heatmaps qualitatively highlight defect region, not background

---

### 7.3 Module 3 — Optical Degradation Simulator

#### Physics Foundation

| Degradation Type | Physics Model | Implementation |
|---|---|---|
| PSF / Diffraction blur | Airy disk: PSF radius ∝ λ/(2·NA). For λ=550nm, NA 0.1–0.5 → σ=1–8px | `cv2.GaussianBlur` with σ derived from NA |
| Depth-of-field blur | Circle of confusion: d = f² · |Δz| / (N · d · (d−f)) | `cv2.blur` with kernel size = f(z_offset) |
| Sensor shot noise | Photon arrival is Poisson; dominant at low light | `np.random.poisson(image * gain) / gain` |
| Read / thermal noise | Gaussian additive, independent of signal | `np.random.normal(0, σ_read, shape)` |
| Illumination non-uniformity | Radial vignetting: I(r) = I₀·cos⁴(α) | Radial mask + linear gradient multiplication |
| MTF / resolution limit | Spatial frequency response falloff at Nyquist | FFT → multiply MTF curve → IFFT |

#### DOE Design — 3-Factor, 3-Level Full Factorial

This is the engineering heart of Bullet 2. A proper DOE produces a quantitative claim about which imaging parameter matters most.

| Factor | Levels | Physical Meaning |
|---|---|---|
| PSF sigma (blur) | Low=1px, Mid=3px, High=6px | NA=0.5, 0.2, 0.1 respectively |
| SNR (sensor noise) | High=40dB, Mid=30dB, Low=20dB | High-end vs mid-range vs budget sensor |
| Illumination uniformity | Uniform, ±10% vignette, ±25% vignette | Well-calibrated vs aging vs misaligned light source |

**27 total configurations (3×3×3).** For each: degrade entire test set (270 images) → classifier inference → record per-class F1. Total inference calls: 27 × 270 = 7,290. Estimated runtime on Colab T4: ~60 seconds.

#### Key Numbers to Extract from DOE

- PSF sigma at which mean F1 first drops below 90% → the **resolution threshold**
- Most sensitive defect class to SNR degradation (expected: fine scratches vs coarse crazing)
- Interaction effect: F1 at PSF=high + SNR=low vs each factor alone
- Main effects plot: rank three factors by F1 impact via OLS in `statsmodels`

#### Implementation Tasks

**Task 3.1 — Degradation functions**
- `apply_psf(img, sigma)`, `apply_noise(img, snr_db)`, `apply_vignette(img, strength)`, `apply_dof_blur(img, z_offset)`
- Unit test: `apply_psf(img, sigma=0)` returns identical image
- Exit criterion: all 4 functions pass unit tests

**Task 3.2 — PSF physics validation**
- Verify: σ_pixels = 0.61·λ / (NA·pixel_size). For pixel_size=5μm, λ=550nm: NA=0.5→σ≈1.3px, NA=0.2→σ≈3.4px, NA=0.1→σ≈6.7px
- Document in README
- Exit criterion: computed σ values match theoretical within 10%

**Task 3.3 — DOE runner**
- `doe_runner.py`: generate 27 combinations with `itertools.product`
- For each: degrade test set → batch inference → record per-class F1 → save to `results/doe_results.csv`
- Exit criterion: CSV has 27×6=162 rows, no NaN values

**Task 3.4 — DOE analysis**
- Main effects plots: 3 subplots, x=factor level, y=mean F1
- Interaction heatmap: PSF×SNR with F1 as color
- `statsmodels` OLS for factor significance
- Exit criterion: can state dominant factor with p<0.05

---

### 7.4 Module 4 — Statistical Process Control

#### SPC Engineering Rationale

SPC is used in semiconductor fabs to detect process drift before it produces out-of-spec product. We simulate this by treating each batch of 25 images as a production lot, computing defect rate statistics, and applying control chart rules to detect abnormal runs.

#### Control Chart Implementation

| Component | Implementation Detail |
|---|---|
| X-bar chart | Track mean defect rate per batch. UCL/LCL = X̄ ± 3σ. Estimated from first 20 baseline batches |
| R chart | Track range of defect rates within batch. Detects variability increase before mean shifts |
| Western Electric Rule 1 | 1 point beyond 3σ → immediate OOC alert |
| Western Electric Rule 2 | 9 consecutive points same side of centerline → trend alert |
| Western Electric Rule 3 | 6 consecutive points monotonically increasing/decreasing → drift alert |
| Western Electric Rule 4 | 2 of 3 consecutive points beyond 2σ → early warning |
| Batch size | 25 images — standard SPC subgroup size; sufficient statistical power |
| Baseline period | First 20 batches (500 images) used to estimate control limits |

#### Synthetic Drift Injection for Validation

Ground truth is required to verify zero false escapes. Created synthetically:

- **Batches 1–20:** draw from clean test set → compute UCL/LCL (baseline)
- **Batches 21–30:** inject +5% additional defects → should trigger OOC within 2–3 batches
- **Batches 31–40:** inject +15% defects + class distribution shift → should trigger immediately
- **False escape rate** = OOC batches not caught / total injected OOC batches → **target: 0%**
- **False alarm rate** = clean batches flagged / total clean batches → **target: <5%**

---

### 7.5 Module 5 — Streamlit Dashboard

| Panel | Content & Interaction |
|---|---|
| Image Upload + Prediction | Upload single image → show preprocessed ROI + predicted class + confidence + Grad-CAM heatmap |
| Batch Inspection | Upload folder (zip) → batch inference → per-class counts, confidence histogram |
| SPC Monitor | Live X-bar and R-charts with UCL/LCL. OOC points in red. Rule violation annotations |
| DOE Results Viewer | Main effects plots, interaction heatmap, resolution threshold marker. Dropdown to filter by class |
| Yield Dashboard | Rolling defect rate, yield % per batch, trend line. Export to CSV button |

#### The ≥80% Manual Review Reduction Claim

- **Baseline:** every frame sent to classifier (100%)
- **Pre-filter:** SSIM vs defect-free reference template → skip frames with SSIM > 0.92
- **Measure:** % frames skipped vs % true defects missed (target: 0 missed)
- **Expected:** ~80% of NEU images are background-dominant → ~80% safely pre-filtered

---

## 8. Tech Stack & Engineering Decisions

| Component | Choice | Alternative | Rationale |
|---|---|---|---|
| DL Framework | PyTorch 2.x | TensorFlow/Keras | Better Grad-CAM ecosystem; more natural custom training loops |
| Model | EfficientNet-B0 (torchvision) | timm library | torchvision has no extra dependency; timm flexibility not needed |
| Augmentation | albumentations | torchvision transforms | 3–10x faster; more industrial transforms (elastic, grid distortion) |
| Image processing | OpenCV 4.x | scikit-image | Industry standard for machine vision; faster morphological ops |
| Optical simulation | NumPy + scipy.ndimage | OpenCV only | scipy gives proper FFT-based MTF filtering; OpenCV lacks frequency domain |
| SPC | Custom (numpy/pandas) | pymc, statsmodels | SPC charts simple enough to implement cleanly; avoids heavy deps |
| DOE analysis | statsmodels OLS | scipy.stats only | Produces proper ANOVA table for main effects — more rigorous |
| Dashboard | Streamlit | Gradio, Dash | Native Plotly charts + file uploader; fastest path to production-quality UI |
| Visualization | Plotly + matplotlib | seaborn | Plotly for interactive dashboard; matplotlib for static DOE report figures |
| Experiment tracking | MLflow (local) | W&B | Local, no account needed; logs training curves + model artifacts |

---

## 9. Developer Tooling — GitHub Copilot & Google Colab Pro

This section defines how the two primary tools are used at each phase and the rules governing their use to keep the work defensible in an interview.

---

### 9.1 GitHub Copilot — Role & Rules

**Role:** Accelerate boilerplate-heavy implementation (DataLoader classes, training loops, Streamlit layout, DOE iteration logic). Copilot handles the scaffolding; you own the engineering decisions.

#### Where Copilot is Used

| Phase | What Copilot Generates | What You Write Manually |
|---|---|---|
| Phase 1 | `NEUDataset` class boilerplate, augmentation Compose pipeline | CLAHE + contour ROI logic — the resume-relevant preprocessing |
| Phase 2 | Training loop skeleton, checkpoint saving, confusion matrix plot | Grad-CAM implementation, grayscale weight adaptation, latency benchmark |
| Phase 3 | `itertools.product` DOE loop, CSV saving boilerplate | Physics derivations for PSF/noise/vignette functions — these must be yours |
| Phase 4 | Streamlit panel layout, Plotly chart wrappers | Western Electric rules logic, SSIM pre-filter threshold tuning |

#### Hard Rules for Copilot Use

- **Never accept Copilot suggestions for physics formulas.** The Airy disk derivation, Poisson noise model, and SPC control limit calculations must be written by you from the equations in this PRD. If an interviewer asks "where does σ = 0.61·λ/NA come from?" you need to own that answer.
- **Never accept Copilot suggestions for the DOE analysis.** The OLS regression and main effects interpretation must be yours.
- **Always read what Copilot generates before accepting.** If you can't explain a line, delete it and write it yourself.
- **Commit message discipline:** use `[copilot]` tag in commits where Copilot wrote >50% of the code in that file, so you can audit later. Example: `feat: add NEUDataset class [copilot]`

#### Copilot Workflow Per Session

```
1. Write the function signature + docstring yourself (defines intent)
2. Let Copilot suggest the body
3. Read every line — reject anything you can't explain
4. Run the function on a small input to verify behavior
5. Only then move to the next function
```

---

### 9.2 Google Colab Pro — Environment & Usage

**Role:** GPU-accelerated training and DOE evaluation. All heavy compute runs here. Local machine handles preprocessing, SPC, dashboard, and light inference benchmarking.

#### Colab Pro vs Pro+ Decision

| Feature | Colab Pro | Colab Pro+ |
|---|---|---|
| GPU access | T4, V100 (queued) | A100 (priority) |
| Background execution | ❌ | ✅ |
| RAM | High RAM available | High RAM |
| Verdict for this project | **Sufficient** — EfficientNet-B0 trains in <20 min on T4 | Only needed if T4 queue is slow |

Use **Colab Pro**. The model is small enough that T4 is plenty. Pro+ is not worth the cost for this project.

#### Colab Notebook Structure

Organize into **four notebooks** — one per heavy compute phase. This prevents the "notebook spaghetti" problem and makes each one shareable/demonstrable independently.

```
notebooks/
├── 01_preprocessing_eda.ipynb       # Phase 1 — dataset audit + preprocessing QA
├── 02_training_evaluation.ipynb     # Phase 2 — EfficientNet training + Grad-CAM
├── 03_optical_doe.ipynb             # Phase 3 — optical simulator + DOE runner
└── 04_spc_validation.ipynb          # Phase 4 — SPC drift injection + validation
```

#### Colab Setup Block (Add to Top of Every Notebook)

```python
# Mount Drive for persistent storage
from google.colab import drive
drive.mount('/content/drive')

# Install dependencies
!pip install -q albumentations opencv-python-headless streamlit plotly mlflow statsmodels

# Set project root
import sys
PROJECT_ROOT = '/content/drive/MyDrive/defect_inspection'
sys.path.insert(0, PROJECT_ROOT)

# Verify GPU
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

#### Data Persistence Strategy

Colab sessions reset. Every artifact that takes >5 minutes to produce must be saved to Google Drive immediately.

| Artifact | Save Location | When to Save |
|---|---|---|
| Preprocessed dataset (cached ROIs) | `drive/defect_inspection/data/preprocessed/` | After Task 1.2 |
| Best model checkpoint (.pth) | `drive/defect_inspection/models/best_model.pth` | After each epoch if val F1 improves |
| Training logs (CSV) | `drive/defect_inspection/logs/training_log.csv` | Every epoch |
| DOE results CSV | `drive/defect_inspection/results/doe_results.csv` | After Task 3.3 |
| Grad-CAM images | `drive/defect_inspection/results/gradcam/` | After Task 2.4 |
| SPC validation results | `drive/defect_inspection/results/spc_validation.json` | After Task 4.2 |

#### GPU Memory Management

EfficientNet-B0 is small but the DOE loop runs 7,290 inferences — memory leaks will crash the session.

```python
# After each DOE configuration — critical
torch.cuda.empty_cache()
import gc; gc.collect()

# Use inference mode for all evaluation (no gradient graph)
with torch.inference_mode():
    outputs = model(batch)

# Mixed precision for training (halves VRAM usage)
with torch.cuda.amp.autocast():
    outputs = model(inputs)
    loss = criterion(outputs, labels)
```

#### Estimated Colab Runtimes

| Task | GPU | Estimated Time |
|---|---|---|
| Phase 1 — preprocessing pipeline | CPU (no GPU needed) | 5–10 min |
| Phase 2 — EfficientNet-B0 training (50 epochs, batch=32) | T4 | 15–20 min |
| Phase 2 — Grad-CAM generation (30 images) | T4 | <2 min |
| Phase 3 — DOE runner (27 configs × 270 images) | T4 | ~60 sec |
| Phase 4 — SPC validation (500 images) | CPU | <2 min |
| **Total GPU time** | | **~25–30 min** |

> Well within Colab Pro's daily GPU allocation. No risk of hitting compute limits.

#### What Runs Locally (Not on Colab)

| Component | Why Local |
|---|---|
| Streamlit dashboard | Requires persistent server; Colab tunneling is unreliable |
| CPU inference benchmarking | Must measure on CPU — Colab would give GPU numbers |
| SSIM pre-filter threshold tuning | Lightweight; no GPU needed |
| Final MLflow experiment review | Local `mlflow ui` is cleaner than Colab port forwarding |

---

### 9.3 Repository Structure

```
defect-inspection/
├── data/
│   ├── raw/                    # NEU dataset (gitignored)
│   └── preprocessed/           # Cached ROI tensors (gitignored)
├── notebooks/
│   ├── 01_preprocessing_eda.ipynb
│   ├── 02_training_evaluation.ipynb
│   ├── 03_optical_doe.ipynb
│   └── 04_spc_validation.ipynb
├── src/
│   ├── preprocessing.py        # M1 — OpenCV pipeline
│   ├── model.py                # M2 — EfficientNet + Grad-CAM
│   ├── optical_simulator.py    # M3 — degradation functions
│   ├── spc.py                  # M4 — SPC monitor
│   └── dataset.py              # PyTorch Dataset class
├── app/
│   └── dashboard.py            # M5 — Streamlit app
├── results/
│   ├── doe_results.csv
│   ├── spc_validation.json
│   └── gradcam/
├── models/
│   └── best_model.pth          # gitignored (too large)
├── requirements.txt
└── README.md
```

---

## 10. Build Phases & Detailed Task Breakdown

> **Total estimated time:** 12–16 focused hours across 2 weekends.
> Each phase has a hard exit criterion. Do not proceed until it is met.

---

### Phase 1 — Data & Preprocessing (3–4 hrs) | Local + Colab Notebook 01

| Task | Detail | Exit Criterion |
|---|---|---|
| 1.1 Dataset setup | Download NEU-DET. Write `dataset_audit.py` to check class balance, image sizes, corrupted files | Audit shows 0 errors, 1,800 images confirmed |
| 1.2 OpenCV pipeline | Implement `preprocess(img)`. Tune CLAHE, adaptive threshold, morphological ops on 20 samples/class | Visual QA pass + ROI hit rate >90% on 180-image sample |
| 1.3 PyTorch Dataset class | `NEUDataset(root, split, transform)` with stratified split. Verify DataLoader shapes | One full batch `(32, 3, 224, 224)` loads without error |
| 1.4 Augmentation pipeline | albumentations Compose. Visualize augmented samples | Augmented images look physically plausible |

---

### Phase 2 — Classifier Training (3–4 hrs) | Colab Notebook 02 (T4 GPU)

| Task | Detail | Exit Criterion |
|---|---|---|
| 2.1 Model setup | EfficientNet-B0 + grayscale weight adaptation + fc replacement | Forward pass on dummy batch completes |
| 2.2 Training loop | Mixed precision, AdamW, cosine LR, early stopping. Log to CSV | 50 epochs, no NaN loss |
| 2.3 Hyperparameter tuning | 3 experiments: LR schedule / label smoothing / augmentation intensity | Val F1 >96% on best config |
| 2.4 Evaluation | Confusion matrix, per-class F1, CPU latency benchmark | Test accuracy ≥97%, CPU latency <8ms |
| 2.5 Grad-CAM | 5 samples/class (30 total), overlay heatmaps | >80% heatmaps qualitatively correct |

---

### Phase 3 — Optical Simulator & DOE (3–4 hrs) | Colab Notebook 03 (T4 GPU)

| Task | Detail | Exit Criterion |
|---|---|---|
| 3.1 Degradation functions | `apply_psf`, `apply_noise`, `apply_vignette`, `apply_dof_blur` with unit tests | All 4 functions pass unit tests |
| 3.2 PSF physics validation | Verify σ values match Airy disk formula within 10%. Document in README | Computed σ matches theory within 10% |
| 3.3 DOE runner | 27 configs, degrade test set, batch inference, save `doe_results.csv` | CSV: 162 rows (27×6), no NaN |
| 3.4 DOE analysis | Main effects plots, interaction heatmap, statsmodels OLS, resolution threshold | Dominant factor identified with p<0.05 |

---

### Phase 4 — SPC & Dashboard (3–4 hrs) | Colab Notebook 04 + Local

| Task | Detail | Exit Criterion |
|---|---|---|
| 4.1 SPC implementation | `SPCMonitor` class with X-bar/R-charts + 4 Western Electric rules | Correct UCL/LCL on synthetic data with known σ |
| 4.2 Drift injection validation | 500-image baseline + injected drift in batches 21–40 | False escape rate = 0%, false alarm rate <5% |
| 4.3 SSIM pre-filter | Compute SSIM vs defect-free reference. Find threshold for 0 missed defects. Measure % skipped | ≥80% frames skipped, 0 missed defects |
| 4.4 Streamlit dashboard | 5 panels, Plotly charts, file uploader, CSV export | All panels render without error on sample data |
| 4.5 Integration test | Full pipeline end-to-end, measure latency | <2 seconds per image, no uncaught exceptions |

---

## 11. Quantification Summary

> Every number in the bullets must be **measured, not estimated.** Fill in after running experiments.

| Metric | Target | How Measured | Bullet |
|---|---|---|---|
| Top-1 accuracy | ≥97% | `sklearn.accuracy_score` on 270-image test set | B1 |
| CPU inference latency | <8ms | `timeit` over 1,000 single-image calls, median | B1 |
| DOE configurations | 27 | 3 factors × 3 levels full factorial | B2 |
| Resolution threshold (PSF σ) | Report actual value | First PSF level where mean F1 < 0.90 | B2 |
| F1 degradation at worst config | Report actual % | Best DOE F1 minus worst DOE F1 | B2 |
| SPC test run size | 500 images | 20 baseline batches × 25 images/batch | B3 |
| False escape rate | 0% | Injected OOC batches missed / total injected | B3 |
| Manual review reduction | ≥80% | % frames skipped by SSIM pre-filter, 0 missed | B3 |

---

## 12. Risks & Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Accuracy target not met (<97%) | Low — NEU is well-studied | More augmentation, longer training. Fallback: report actual number — 94–95% is still strong |
| CPU latency >8ms | Medium — hardware dependent | Export to ONNX (2–3x CPU speedup). Fallback: report GPU latency + note ONNX path |
| DOE shows no significant factor | Very Low — PSF effect is well-established | Report raw F1 curves regardless. Visual trend is interview-defensible even without p<0.05 |
| SSIM pre-filter misses defects | Medium | Tune threshold conservatively. Fallback: report 70% reduction if needed |
| Colab session resets mid-training | Medium | Checkpoint every epoch to Drive. Resume from best checkpoint |
| Copilot generates wrong physics formula | Medium | Never accept Copilot for physics. Write all formulas manually from PRD equations |
| SPC false escapes >0 | Low | Use 2.5σ instead of 3σ for tighter limits. Increases false alarms but guarantees zero escapes |

---

## 13. Final Resume Bullets — Post-Build

> Fill in bracketed values after running experiments. These paste directly into your LaTeX resume.

**B1:**
> Engineered an **OpenCV + EfficientNet-B0** defect classification pipeline on the NEU Surface Defect dataset achieving **[X]% top-1 accuracy** across 6 defect classes with **[Y]ms CPU inference latency** per frame, using CLAHE normalization, adaptive thresholding, and contour-guided ROI extraction.

**B2:**
> Designed an **optical degradation simulator** modeling PSF convolution (Airy disk, NA 0.1–0.5), Poisson + Gaussian sensor noise, and illumination vignetting; ran a **27-configuration DOE** (3-factor × 3-level), identifying PSF σ=[Z]px as the resolution threshold beyond which F1 degrades by **[W]%**, informing camera/lens selection trade-offs for high-volume inspection.

**B3:**
> Implemented **Statistical Process Control** (X-bar/R-charts, 4 Western Electric rules) over simulated production batches achieving **zero false escapes** across 500 images; deployed SSIM-based frame pre-filter reducing classifier load by **[V]%** with zero missed defects; surfaced Grad-CAM heatmaps, OOC alerts, and batch yield in a **Streamlit dashboard**.

---

> *Build what matters. Measure everything. Ship the bullets.*
