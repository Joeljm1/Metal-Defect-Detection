# Project Modules & Implementation Status (Final — 100% Complete)
## Project: Computer Vision Based Detection of Surface Defects in Metal Components
**Course:** CSE411 Computer Vision | **Team Number:** 6

---

### Team Members:
- **Joel Joseph Mathews** (`2023BCS0061`)
- **Karthik Das P** (`2023BCS0058`)
- **Vivek Binod** (`2023BCS0043`)
- **Vibhaas Nirantar Srivastava** (`2023BCS0037`)

---

## 1. Overview of Project Lifecycle

```
[Completed in Task 4 (~25%)]
├── 1. Dataset Collection & Ingestion (NEU-DET 1,800 images)
├── 2. CLAHE & Bilateral Filtering Preprocessing Pipeline
├── 3. ECA & Spatial Attention Module Architecture
├── 4. CSPDarknet Backbone + PANet Neck + Multi-Scale Head
├── 5. Controlled Ablation Framework (M1, M2, M3, M4)
└── 6. Engineering Infrastructure (uv, make, pytest, EDA)

[Completed in Tasks 5-7 (remaining 75%)]
├── Phase 2: Full-Scale Ablation Model Training & Optimization -- DONE (Task 5)
│   ├── Multi-epoch training of M1, M2, M3, M4 (checkpoints/)
│   ├── Loss convergence analysis
│   └── IoU-distance K-Means anchor clustering (k=9)
├── Phase 3: Explainability & Cross-Dataset Generalization -- DONE (Task 5-6)
│   ├── Grad-CAM / Eigen-CAM visual saliency heatmaps
│   └── Cross-domain evaluation on GC10-DET (Hypothesis H4)
└── Phase 4: Optimization, Deployment & Interactive Demonstration -- DONE (Task 6-7)
    ├── ONNX export + multi-format edge benchmark (116.1-219.3 FPS)
    ├── Interactive Streamlit inspection dashboard (app.py)
    └── Final research report & oral presentation (Task 7)
```

---

## 2. Module Status

### Module 1: Full-Scale Model Convergence & Training (Phase 2) — COMPLETED (Task 5)
* **Objective:** Train all 4 models in the Controlled Ablation Matrix (M1 to M4) to full convergence over 50–100 epochs.
* **Implementation:**
  1. Config-driven training entry points: `scripts/train.py`, `scripts/train_ablation.py`.
  2. Deterministic 3-way train/val/test partitioning; CIoU box regression loss (`src/training/loss.py`, `src/training/trainer.py`).
  3. Loss convergence curves via `scripts/plot_loss_curves.py`; top-performing weights checkpointed in `checkpoints/`.

### Module 2: Anchor Box Optimization & Hyperparameter Tuning (Phase 2) — COMPLETED (Task 5)
* **Objective:** Adapt default YOLO anchor boxes to the distribution of metallic defect aspect ratios.
* **Implementation:**
  1. IoU-distance K-Means clustering ($k=9$) over NEU-DET bounding box dimensions: `scripts/cluster_anchors.py`.
  2. CLAHE parameter selection on the CIELAB $L$-channel ($\text{clip\_limit} \in [1.5, 3.0]$, $\text{tile\_grid} \in [4\times 4, 8\times 8]$): `src/preprocessing/clahe.py`, `configs/`.

### Module 3: Grad-CAM Saliency Interpretability Module (Phase 3) — COMPLETED (Task 5)
* **Objective:** Provide explainable visual feedback to plant operators indicating which image regions triggered defect classifications.
* **Implementation:**
  1. Multi-scale Grad-CAM attribution hooked into `c3_fpn2`, `c3_pan1`, and `att_p3` neck layers (`src/evaluation/gradcam.py`):
     $$\alpha_k^c = \frac{1}{Z} \sum_i \sum_j \frac{\partial y^c}{\partial A_{i,j}^k}$$
     $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$
  2. Heatmap rendering overlaid on original metallic surfaces alongside predicted bounding boxes (`src/utils/visualization.py`, `scripts/visualize_gradcam.py`).
  3. 24-panel explainability gallery validating sharper attention concentration on subtle defect morphology.

### Module 4: Cross-Dataset Generalization on GC10-DET (Phase 3) — COMPLETED (Task 6)
* **Objective:** Test Hypothesis H4 (Cross-domain robustness across differing industrial imaging setups).
* **Implementation:**
  1. Zero-shot cross-domain mapping and few-shot linear probing with frozen backbone/neck feature extractors (`src/evaluation/domain_adaptation.py`, `scripts/evaluate_gc10.py`).
  2. **Result:** Few-shot transfer on 10 GC10-DET defect classes reaches **17.48% mAP@0.5** for M4 vs **11.50% mAP@0.5** for baseline M1 (**+52.0% relative gain**).

### Module 5: Latency Optimization & ONNX Runtime Export (Phase 4) — COMPLETED (Task 6)
* **Objective:** Fulfil the Hypothesis H3 requirement of maintaining real-time edge throughput ($\ge 30\text{-}50$ FPS).
* **Implementation:**
  1. ONNX export with dynamic batch dimension and full graph validation via `onnx.checker` (`src/deployment/export.py`, `scripts/export_and_benchmark_onnx.py`); uses the `torch.export`-based ONNX exporter.
  2. FP16 half-precision export and multi-format benchmarking across PyTorch FP32/FP16 and ONNX Runtime CPU/CUDA (`src/evaluation/benchmark.py`).
  3. **Result:** 116.1–219.3 FPS across formats — exceeds the industrial edge threshold by $>3.8\times$.

### Module 6: Interactive End-to-End Demonstration Dashboard (Phase 4) — COMPLETED (Task 7)
* **Objective:** Deliver a practical inspection demonstration for manufacturing quality control.
* **Implementation:** Streamlit plant-operator web application (`app.py`, `src/deployment/dashboard.py`, `scripts/run_dashboard.py`):
  - Model selection across all four ablation models (M1–M4) with live latency/FPS telemetry.
  - Image upload or sample dataset selection; confidence and NMS IoU sliders.
  - Four-panel view: Raw Input | CLAHE + Bilateral Enhanced | Bounding-Box Detections | Grad-CAM Saliency.
  - Downloadable audit report.

---

## 3. Verification of Research Hypotheses (Final Results)

| Hypothesis | Empirical Validation & Results | Status |
|---|---|:---:|
| **H1 (Contrast Normalization)** | On low-contrast directional *Scratches*, M4 doubles baseline AP from **12.68% to 25.87%** (**+104% relative**). | **CONFIRMED** |
| **H2 (Attention Gating)** | ECA + Spatial Attention add only **309 parameters** ($<0.005\%$ overhead), boosting *Inclusions* from **37.50% to 47.26% AP** (**+26.0% relative**). | **CONFIRMED** |
| **H3 (Real-Time Edge Viability)** | M4 achieves **219.3 FPS** (4.56 ms) in PyTorch and **116.1 FPS** (8.61 ms) in ONNX Runtime CPU — **$>3.8\times$** the $\ge 30\text{-}50$ FPS threshold. | **CONFIRMED** |
| **H4 (Cross-Dataset Generalizability)** | Few-shot GC10-DET transfer: M4 **17.48% mAP@0.5** vs M1 **11.50% mAP@0.5** (**+52.0% relative**). | **CONFIRMED** |

---

## 4. Automated Verification

* **40 passing unit and integration tests** (`make test`, pytest) covering dataset parsing, preprocessing, models, attention, loss, training, evaluation, Grad-CAM, domain adaptation, ONNX export, and the dashboard.
* ONNX graphs validated with `onnx.checker`; exports verified for dynamic batch execution in ONNX Runtime.

---

## 5. Work Allocation & Timeline

| Team Member | Primary Module Responsibility | Secondary Support |
|---|---|---|
| **Joel Joseph Mathews** (`2023BCS0061`) | Preprocessing Pipeline, Architecture Integration & Model M4 | UI Demonstration & Final Report |
| **Karthik Das P** (`2023BCS0058`) | Attention Modules (ECA & Spatial), Training M3 | Benchmark & Latency Profiling |
| **Vivek Binod** (`2023BCS0043`) | Loss Functions, CIoU Regression, Training M1 & M2 | EDA & Anchor Box Optimization |
| **Vibhaas Nirantar Srivastava** (`2023BCS0037`) | Evaluation Metrics, mAP Harness & GC10-DET Transfer | Grad-CAM Saliency Maps |
