# Remaining Project Modules & Implementation Roadmap (Remaining 75%)
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

[Remaining to be Completed (~75%)]
├── Phase 2: Full-Scale Ablation Model Training & Optimization (~25%)
│   ├── Multi-epoch training of M1, M2, M3, M4 on GPU
│   ├── Loss convergence analysis and tensorboard logging
│   └── Hyperparameter tuning (k-means anchor clustering, SIoU/CIoU loss weights)
├── Phase 3: Explainability & Cross-Dataset Generalization (~25%)
│   ├── Grad-CAM / Eigen-CAM visual saliency heatmap generator
│   ├── Cross-domain evaluation on GC10-DET metallic defect benchmark
│   └── Statistical hypothesis testing (H1, H2, H3, H4)
└── Phase 4: Optimization, Deployment & Interactive Demonstration (~25%)
    ├── FP16 / ONNX runtime latency optimization (>= 30-50 FPS target)
    ├── Interactive Web Demonstration UI (Streamlit / Gradio)
    └── Final Research Report & Oral Presentation
```

---

## 2. Detailed Technical Breakdown of Remaining Modules

### Module 1: Full-Scale Model Convergence & Training (Phase 2)
* **Objective:** Train all 4 models in the Controlled Ablation Matrix (M1 to M4) to full convergence over 50–100 epochs.
* **Tasks:**
  1. Train **M1** (Baseline YOLOv5s) to establish the experimental reference point ($\approx 75.1\% - 81.6\%\text{ mAP}$).
  2. Train **M2** (Baseline + CLAHE/Bilateral Preprocessing) to isolate the empirical gain from contrast normalization (Hypothesis H1).
  3. Train **M3** (Baseline + ECA & Spatial Attention) to quantify feature preservation on subtle defects (Hypothesis H2).
  4. Train **M4** (Proposed Integrated Model) to evaluate the combined synergy.
  5. Checkpoint the top-performing model weights in `checkpoints/`.

### Module 2: Anchor Box Optimization & Hyperparameter Tuning (Phase 2)
* **Objective:** Adapt default YOLO anchor boxes to the distribution of metallic defect aspect ratios.
* **Tasks:**
  1. Implement K-means clustering on NEU-DET bounding box dimensions to derive 9 custom anchor dimensions matching small pits vs elongate scratches.
  2. Perform grid search for optimal CLAHE parameters ($\text{clip\_limit} \in [1.5, 3.0]$, $\text{tile\_grid} \in [4\times 4, 8\times 8]$).

### Module 3: Grad-CAM Saliency Interpretability Module (Phase 3)
* **Objective:** Provide explainable visual feedback to plant operators indicating which image regions triggered defect classifications.
* **Tasks:**
  1. Compute gradient of defect classification score $y^c$ with respect to feature activation maps $A^k$ in the P3/P4/P5 detection neck:
     $$\alpha_k^c = \frac{1}{Z} \sum_i \sum_j \frac{\partial y^c}{\partial A_{i,j}^k}$$
     $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$
  2. Render heatmaps overlaid on original metallic surfaces alongside predicted bounding boxes.
  3. Generate comparative heatmaps showing how attention modules focus more sharply on subtle crazing vs baseline diffuse activations.

### Module 4: Cross-Dataset Generalization on GC10-DET (Phase 3)
* **Objective:** Test Hypothesis H4 (Cross-domain robustness across differing industrial imaging setups).
* **Tasks:**
  1. Evaluate trained NEU-DET detectors on held-out GC10-DET metallic surface defects without fine-tuning (zero-shot transfer).
  2. Fine-tune on GC10-DET few-shot split and compare transfer efficiency between M1 (baseline) and M4 (proposed).
  3. Quantify performance drop $\Delta\text{mAP} = \text{mAP}_{\text{NEU}} - \text{mAP}_{\text{GC10}}$.

### Module 5: Latency Optimization & ONNX Runtime Export (Phase 4)
* **Objective:** Fulfill Hypothesis H3 requirement of maintaining real-time edge throughput ($\ge 30-50\text{ FPS}$).
* **Tasks:**
  1. Export PyTorch weights to ONNX format with dynamic batch axes.
  2. Benchmark FP16 half-precision and TensorRT / ONNXRuntime execution.
  3. Profile memory footprint and per-layer FLOPs.

### Module 6: Interactive End-to-End Demonstration Dashboard (Phase 4)
* **Objective:** Deliver a practical inspection demonstration for manufacturing quality control.
* **Tasks:**
  1. Build a Streamlit or Gradio interactive web UI.
  2. Features:
     - Image upload or camera stream.
     - Toggleable preprocessing views (Raw vs CLAHE vs Bilateral).
     - Bounding box detections with class probabilities and confidence sliders.
     - Grad-CAM heatmap visualization toggle.
     - Real-time latency and FPS telemetry display.

---

## 3. Work Allocation & Timeline

| Team Member | Primary Module Responsibility | Secondary Support |
|---|---|---|
| **Joel Joseph Mathews** (`2023BCS0061`) | Preprocessing Pipeline, Architecture Integration & Model M4 | UI Demonstration & Final Report |
| **Karthik Das P** (`2023BCS0058`) | Attention Modules (ECA & Spatial), Training M3 | Benchmark & Latency Profiling |
| **Vivek Binod** (`2023BCS0043`) | Loss Functions, CIoU Regression, Training M1 & M2 | EDA & Anchor Box Optimization |
| **Vibhaas Nirantar Srivastava** (`2023BCS0037`) | Evaluation Metrics, mAP Harness & GC10-DET Transfer | Grad-CAM Saliency Maps |
