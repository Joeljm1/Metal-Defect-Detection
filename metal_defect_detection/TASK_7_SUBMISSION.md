# CSE411 Computer Vision — Project Task 7 Submission
## Implementation — Part 4: 100% Final Overall Milestone

---

### Submission Metadata
* **Course:** CSE411 Computer Vision
* **Project Title:** Computer Vision Based Detection of Surface Defects in Metal Components
* **Team Number:** 6
* **Team Members & Roll Numbers:**
  1. **Joel Joseph Mathews** — `2023BCS0061`
  2. **Karthik Das P** — `2023BCS0058`
  3. **Vivek Binod** — `2023BCS0043`
  4. **Vibhaas Nirantar Srivastava** — `2023BCS0037`

---

## 1. Executive Summary & 100% Final Implementation Milestone

In accordance with the **Project Task 7** specifications:
> *"Complete the entire proposed methodology. Integrate and test all modules. Submit the final report with the source code."*

Our team has completed **100% of the proposed project methodology**, delivering a fully functional, robustly tested, and deployment-ready industrial computer vision system for metallic surface defect detection.

### Project Lifecycle Completion Matrix (100% Complete):

| Project Phase | Focus Area | Technical Deliverables & Implementation | Status |
| :---: | :--- | :--- | :---: |
| **Phase 1** | **Dataset & Preprocessing Pipeline** | Ingestion of 1,800 NEU-DET images and VOC XML annotations; CIELAB L-channel CLAHE contrast normalization; edge-preserving Bilateral filtering; dynamic geometric data augmentation. | **100% Completed** |
| **Phase 2** | **Neural Architecture & Loss Formulation** | CSPDarknet backbone; PANet feature pyramid neck with ECA channel attention and Spatial Attention modules; multi-task CIoU bounding box regression loss; IoU-distance K-Means anchor clustering ($k=9$). | **100% Completed** |
| **Phase 3** | **Multi-Epoch Convergence Training & Ablation** | Multi-epoch convergence training across the controlled ablation matrix (M1–M4); deterministic 3-way train/val/test data partitioning; evaluation protocol with full-curve mAP@0.5 and operating point $P/R/F1$. | **100% Completed** |
| **Phase 4** | **Explainable AI (Grad-CAM Saliency)** | Multi-scale Grad-CAM attribution hooked into `c3_fpn2`, `c3_pan1`, and `att_p3` neck layers; 24-panel explainability gallery validating attention concentration on defect morphology. | **100% Completed** |
| **Phase 5** | **Domain Generalization & Edge Export** | Cross-dataset domain adaptation on GC10-DET across 10 defect classes (Hypothesis H4 confirmed with $+52.0\%$ gain); ONNX model graph export; multi-format edge runtime benchmarking ($116.1$ to $183.7$ FPS). | **100% Completed** |
| **Phase 6** | **Interactive Inspection Web Dashboard** | Streamlit-based plant operator web application (`app.py`); live image upload; adjustable confidence/IoU sliders; side-by-side preprocessing; single-click Grad-CAM toggle; downloadable audit report. | **100% Completed** |

---

## 2. Verification of Research Hypotheses

| Hypothesis | Theoretical Principle | Empirical Validation & Results | Status |
| :--- | :--- | :--- | :---: |
| **Hypothesis H1 (Contrast Normalization)** | CLAHE on CIELAB $L$-channel and bilateral filtering normalize non-uniform factory illumination and sharpen defect boundaries. | On low-contrast directional *Scratches*, M4 doubles baseline accuracy from **12.68% to 25.87% AP** (**+104% relative gain**). | **CONFIRMED** |
| **Hypothesis H2 (Attention Gating)** | Efficient Channel Attention (ECA) and Spatial Attention suppress steel surface reflection noise with minimal parameter overhead. | Attention modules add only **309 parameters** ($<0.005\%$ overhead), boosting *Inclusions* from **37.50% to 47.26% AP** (**+26.0% relative gain**). | **CONFIRMED** |
| **Hypothesis H3 (Real-Time Edge Viability)** | The lightweight integrated detector sustains real-time throughput ($\ge 30-50$ FPS) on embedded edge compute. | M4 achieves **219.3 FPS** (4.56 ms latency) in PyTorch and **116.1 FPS** (8.61 ms latency) in ONNX Runtime CPU, exceeding threshold by **$>3.8\times$**. | **CONFIRMED** |
| **Hypothesis H4 (Cross-Dataset Generalizability)** | The contrast-enhanced attention representation exhibits superior transferability to unseen manufacturing domains (GC10-DET). | Under few-shot adaptation on 10 GC10-DET defect classes, M4 achieves **17.48% mAP@0.5** vs **11.50% mAP@0.5** for M1 (**+52.0% relative gain**). | **CONFIRMED** |

---

## 3. Interactive Plant Operator Dashboard (`app.py`)

To deliver a practical, deployment-ready quality control prototype for factory manufacturing lines, we implemented an interactive web application built with Streamlit (`app.py`, `src/deployment/dashboard.py`).

### Key Dashboard Capabilities:
1. **Model Selection & Telemetry:** Allows operators to select between all four trained ablation models (M1 to M4) and displays live telemetry cards: total frame latency, preprocessing time, inference time, frame rate (FPS), and active hardware acceleration device.
2. **Defect Inspection Sliders:** Real-time adjustment of detection confidence threshold ($0.05 - 0.95$) and NMS IoU threshold ($0.10 - 0.80$).
3. **Four-Panel Visual Inspection View:**
   - *Panel 1 (Raw Input):* Displays original metallic surface image uploaded by operator or selected from sample dataset.
   - *Panel 2 (Enhanced View):* Displays CIELAB CLAHE + Bilateral preprocessed image highlighting localized edge gradients.
   - *Panel 3 (Localized Defect Detections):* Displays predicted bounding boxes color-coded by defect class with confidence percentages.
   - *Panel 4 (Grad-CAM Explainable Saliency):* Displays overlay heatmaps with selectable colormaps (JET, VIRIDIS, HOT, INFERNO) and opacity blending slider.
4. **Audit Log & Report Export:** Tabulates detected defect coordinates, class classifications, and confidence values, with a single-click button to download structured JSON audit reports (`defect_inspection_report.json`).

Visual demonstration captured in `reports/figures/dashboard_inspection_demo.png`.

---

## 4. Complete Software Architecture & Codebase Inventory

### Source Modules (`src/`):
* `src/dataset/parser.py`: NEU-DET VOC XML parsing and ground-truth bounding box coordinate transformation.
* `src/dataset/loader.py`: Deterministic 3-way data partitioner (`train`, `val`, `test`), dynamic flips, and PyTorch DataLoaders.
* `src/dataset/eda.py`: Exploratory data analysis, aspect ratio distributions, and defect spatial frequency calculations.
* `src/preprocessing/clahe.py`: CIELAB color-space CLAHE contrast normalization.
* `src/preprocessing/bilateral.py`: Edge-preserving bilateral filtering and noise smoothing.
* `src/preprocessing/pipeline.py`: Composite `DefectPreprocessor` pipeline combining CLAHE and bilateral filtering.
* `src/models/backbone.py`: CSPDarknet feature extraction backbone with cross-stage partial connections.
* `src/models/neck.py`: PANet multi-scale feature pyramid neck with bottom-up aggregation and attention blocks.
* `src/models/attention.py`: 1D convolution Efficient Channel Attention (ECA) and Spatial Attention modules.
* `src/models/head.py`: Multi-scale detection head with dynamic anchor grid decoding.
* `src/models/detector.py`: Master `DefectDetector` assembly with modular ablation configurations (M1–M4).
* `src/training/loss.py`: Multi-task loss combining CIoU box regression, unweighted objectness BCE, and class BCE.
* `src/training/trainer.py`: Full convergence training engine with Cosine Annealing learning rate schedule.
* `src/evaluation/metrics.py`: VOC-style 101-point AP interpolation, full-curve mAP@0.5, and operating point $P/R/F1$.
* `src/evaluation/benchmark.py`: High-precision inference latency and FPS throughput benchmark harness.
* `src/evaluation/gradcam.py`: Multi-scale Grad-CAM visual attribution engine hooked into neck fusion layers.
* `src/evaluation/domain_adaptation.py`: GC10-DET dataset loader, morphological semantic mapping, and few-shot linear probing.
* `src/deployment/export.py`: ONNX model exporter with dynamic batching, graph validation, and edge runtime profiling.
* `src/deployment/dashboard.py`: Streamlit dashboard helper functions, color schemes, and defect inspection pipeline.
* `src/utils/box_ops.py`: IoU, CIoU, box format conversions, and class-aware non-maximum suppression.
* `src/utils/visualization.py`: Image annotation utilities and bounding box visualizers.

### Standalone CLI Scripts (`scripts/`):
* `scripts/cluster_anchors.py`: IoU-distance K-Means anchor box clustering for metallic defects ($k=9$).
* `scripts/train_ablation.py`: Multi-model convergence training runner for M1–M4 ablation matrix.
* `scripts/plot_loss_curves.py`: Plots training and validation loss convergence trajectories.
* `scripts/visualize_gradcam.py`: Generates 24-panel multi-class Grad-CAM visual explainability gallery.
* `scripts/evaluate_gc10.py`: Executes GC10-DET cross-dataset domain adaptation benchmark.
* `scripts/export_and_benchmark_onnx.py`: Exports PyTorch models to ONNX and benchmarks edge runtime engines.
* `scripts/generate_dashboard_demo.py`: Generates multi-panel inspection demonstration figure for dashboard.
* `scripts/run_dashboard.py`: CLI launcher for the interactive Streamlit web application.
* `scripts/benchmark_fps.py`: Profiles model forward latency and end-to-end FPS across 100 iterations.
* `scripts/evaluate.py`: Standalone CLI checkpoint evaluator.

### Test Suite:
* `tests/`: **39 passing unit and integration tests** verifying 100% of software components.

---

## 5. Automated Verification & Test Status

All **39 tests pass** cleanly in `pytest`:
```bash
uv run pytest tests/ -v
# 39 passed, 6 warnings in 12.43s
```
Test suite breakdown:
* `test_attention.py`: ECA channel & spatial attention tensor operations (3 tests)
* `test_benchmark.py`: Hardware benchmark timing and FPS validation (1 test)
* `test_dashboard.py`: Model caching, pipeline inspection, and color mapping (3 tests)
* `test_dataset.py`: VOC parsing, YOLO collate, and 3-way split determinism (4 tests)
* `test_domain_adaptation.py`: GC10 loading, remapping, and few-shot adaptation (4 tests)
* `test_evaluation_script.py`: Evaluation script execution smoke test (1 test)
* `test_export.py`: ONNX export, session creation, and edge profiling (3 tests)
* `test_gradcam.py`: Multi-layer hooks, CAM normalization, and JET overlays (4 tests)
* `test_loss_metrics.py`: CIoU regression, unweighted BCE, full-curve mAP, operating $P/R$ (7 tests)
* `test_models.py`: CSPDarknet, PANet neck, and ablation model forward graphs (3 tests)
* `test_preprocessing.py`: CLAHE equalization, bilateral filtering, composite pipeline (3 tests)
* `test_trainer.py`: Single step optimization and checkpointing (1 test)
* `test_visualization.py`: Box annotation drawing utilities (2 tests)

---

## 6. Conclusion

With Task 7, the proposed computer vision system for metal surface defect detection is **100% implemented, integrated, and verified**. The system unifies edge-preserving contrast normalization (CLAHE + Bilateral), lightweight multi-scale attention (ECA + Spatial Attention), explainable visual attribution (Grad-CAM), robust cross-plant generalizability (GC10-DET transfer), low-power edge runtime deployment (ONNX), and an intuitive plant operator dashboard.
