# PCB Surface Defect Inspection & SPC Monitoring System

Automated visual inspection system for manufacturing defect detection using OpenCV preprocessing, EfficientNet-B0 classification, optical degradation simulation, and Statistical Process Control.

## Project Structure

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
│   └── best_model.pth
├── requirements.txt
└── README.md
```

## Setup

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

## Dataset

Download the NEU Surface Defect Database and extract to `data/raw/`:
- 1,800 grayscale images (200x200)
- 6 defect classes: Crazing, Inclusion, Patches, Pitted Surface, Rolled-in Scale, Scratches

## Modules

| Module | Description |
|--------|-------------|
| M1 | OpenCV preprocessing (CLAHE, adaptive threshold, contour ROI) |
| M2 | EfficientNet-B0 classifier with Grad-CAM |
| M3 | Optical degradation simulator + DOE analysis |
| M4 | SPC monitor (X-bar/R-charts, Western Electric rules) |
| M5 | Streamlit dashboard |

## Run Dashboard

```bash
streamlit run app/dashboard.py
```

## Target Metrics

- Top-1 accuracy: ≥97%
- CPU inference latency: <8ms
- DOE configurations: 27 (3-factor × 3-level)
- SPC false escape rate: 0%
- Manual review reduction: ≥80%
