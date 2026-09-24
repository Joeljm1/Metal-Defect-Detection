# Computer Vision Based Detection of Surface Defects in Metal Components
**CSE411 Computer Vision — Course Project (Team 6)**

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Managed by uv](https://img.shields.io/badge/Managed%20by-uv-purple.svg)](https://github.com/astral-sh/uv)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg)](https://pytorch.org/)

---

## 👥 Team Information
* **Team Number:** 6
* **Team Members:**
  - **Joel Joseph Mathews** — `2023BCS0061`
  - **Karthik Das P** — `2023BCS0058`
  - **Vivek Binod** — `2023BCS0043`
  - **Vibhaas Nirantar Srivastava** — `2023BCS0037`

---

## 📌 Project Overview
Automated visual inspection of metallic components in industrial manufacturing lines. This repository implements an end-to-end computer vision pipeline combining **CLAHE illumination and contrast normalization**, **edge-preserving bilateral filtering**, and **lightweight multi-scale attention (ECA + Spatial Attention)** built on a CSPDarknet backbone to detect subtle, low-contrast surface defects (crazing, scratches, pits, patches, inclusions, rolled-in scale).

---

## 📁 Repository Structure
```
.
├── Makefile                      # Automated build, test, and training targets
├── pyproject.toml                # uv package specifications & dependencies
├── .python-version               # Pinned Python 3.12
├── TASK_4_SUBMISSION.md          # Comprehensive Task 4 submission report
├── TASK_5_SUBMISSION.md          # Task 5 submission report (50% milestone)
├── CODEBASE_AUDIT.md             # Codebase audit & methodology review
├── REMAINING_MODULES.md          # Roadmap for the remaining project modules
├── configs/                      # YAML configurations
│   ├── dataset.yaml              # NEU-DET and GC10-DET dataset specifications
│   ├── preprocessing.yaml        # CLAHE and bilateral filtering parameters
│   ├── model_m1_baseline.yaml    # M1 Baseline (vanilla YOLOv5s)
│   ├── model_m2_clahe.yaml       # M2 Baseline + CLAHE/Bilateral Preprocessing
│   ├── model_m3_attention.yaml   # M3 Baseline + ECA/Spatial Attention
│   └── model_m4_integrated.yaml  # M4 Proposed Integrated Architecture
├── data/
│   └── NEU-DET/                  # 1,800 images & YOLO bounding box labels
├── src/                          # Modular source package
│   ├── dataset/                  # Dataset parser, PyTorch loader, and EDA
│   ├── preprocessing/            # CLAHE, Bilateral filtering, and enhancement pipeline
│   ├── models/                   # CSPDarknet backbone, PANet neck, ECA/Spatial attention, head
│   ├── training/                 # CIoU loss and trainer loop
│   ├── evaluation/               # Precision, Recall, F1, mAP@0.5, and FPS benchmarks
│   └── utils/                    # Bounding box ops, NMS, and visualization
├── scripts/                      # Executable CLI scripts
│   ├── download_dataset.py       # Dataset verification & download
│   ├── run_eda.py                # Exploratory data analysis CLI
│   ├── preprocess_dataset.py     # Batch dataset enhancement CLI
│   ├── visualize_preprocessing.py# Preprocessing comparison plot generator
│   ├── cluster_anchors.py        # K-means anchor box clustering (k=9)
│   ├── benchmark_fps.py          # Model latency & FPS benchmarking
│   ├── train.py                  # Training CLI supporting M1-M4 ablation models
│   ├── train_ablation.py         # Full M1-M4 convergence training runner
│   ├── evaluate.py               # Test-split evaluation (P/R/F1, mAP@0.5)
│   ├── visualize_gradcam.py      # Grad-CAM interpretability gallery generator
│   └── plot_loss_curves.py       # Training convergence plot generator
├── reports/                      # EDA reports, statistics JSON, and figures
└── tests/                        # Comprehensive pytest test suite
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites
- Linux / macOS / Windows with Bash
- [`uv`](https://github.com/astral-sh/uv) package manager (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- `make`

### 2. Environment Setup
Install dependencies with a single command using `uv` and `make`:
```bash
make install
```

### 3. Verify Dataset
```bash
make setup-data
```

### 4. Run Exploratory Data Analysis (EDA)
Calculates class distributions, bounding box areas, aspect ratios, and generates summary charts:
```bash
make eda
```

### 5. Preprocess Dataset (CLAHE + Bilateral)
Executes the image enhancement pipeline across all training and test images:
```bash
make preprocess
```

### 6. Generate Preprocessing Comparison Visuals
Creates side-by-side visual comparisons (Raw vs Bilateral vs CLAHE vs Enhanced + Sobel edge map):
```bash
make visualize
```

### 7. Run Unit Tests
```bash
make test
```

### 8. Benchmark Latency & FPS (Hypothesis H3)
```bash
make benchmark
```

### 9. Sanity Training Dry-Run
```bash
make dry-run
```

---

## 🔬 Controlled Ablation Study Matrix

Measured results from the Task 5 run (15 epochs, RTX 5050) — to be regenerated
under the corrected training/evaluation harness (see "Training & Evaluation
Policy") after full 50–100 epoch convergence training:

| Model | CLAHE Preprocessing | ECA / Spatial Attention | mAP@0.5 (Task 5) | E2E FPS |
|---|:---:|:---:|:---:|:---:|
| **M1: Baseline** | ❌ | ❌ | 0.3978 | 310.1 |
| **M2: Preprocessing** | ✅ | ❌ | 0.3747 | 234.0 |
| **M3: Attention** | ❌ | ✅ | 0.3552 | 285.2 |
| **M4: Proposed Integrated**| ✅ | ✅ | **0.4163** | 219.3 |

All variants exceed the industrial real-time requirement ($\ge 30$ FPS end-to-end).

Run any ablation model:
```bash
make train-m1   # Baseline
make train-m2   # Baseline + Preprocessing
make train-m3   # Baseline + Attention
make train-m4   # Proposed Integrated Model
```

---

## 🧪 Training & Evaluation Policy

* **Data splits:** the training split (1,620 images) is carved deterministically
  into train + validation (10%, `split_seed=42`) by `create_dataloaders`.
  Checkpoints are selected on the **validation** split; the **test** split
  (180 images) is evaluated exactly once at the end.
* **Evaluation thresholds:** mAP@0.5 is computed over the full confidence-ranked
  curve (NMS at `conf=0.001`); Precision / Recall / F1 are reported at the fixed
  operating point `conf=0.25` (see `src/evaluation/metrics.py`).
* **Augmentation:** training uses random horizontal/vertical flips (p=0.5 each)
  on the training split. Disable with `--no-augment` in `scripts/train.py` for
  exact reproducibility of published tables.
* **Objectness loss:** standard BCE (`obj_pos_weight=1.0`, i.e. no class
  re-weighting — the stock YOLOv5 operating regime).
