# CSE411 Computer Vision — Project Task 4 Submission
## Implementation — Part 1

---

### Submission Metadata
* **Project Title:** Computer Vision Based Detection of Surface Defects in Metal Components
* **Team Number:** 6
* **Team Members & Roll Numbers:**
  1. **Joel Joseph Mathews** — `2023BCS0061`
  2. **Karthik Das P** — `2023BCS0058`
  3. **Vivek Binod** — `2023BCS0043`
  4. **Vibhaas Nirantar Srivastava** — `2023BCS0037`

---

## 1. Executive Summary & Deliverables Completed
For Project Task 4 (Implementation – Part 1), our team has delivered:
1. **Dataset Collection & Verification:** 100% complete collection of the **NEU-DET** (Northeastern University Surface Defect Database) containing 1,800 high-resolution steel surface images across 6 defect classes in standard YOLO annotation format.
2. **Preprocessing & Image Enhancement Pipeline:** Fully implemented and validated **CLAHE** (Contrast-Limited Adaptive Histogram Equalization) and **Bilateral Filtering** (edge-preserving denoising) to mitigate non-uniform industrial illumination and preserve fine micro-crack boundaries.
3. **25% Methodology Implementation:**
   - Designed and coded the **Lightweight ECA (Efficient Channel Attention)** and **Spatial Attention Modules** to preserve faint defect gradients.
   - Built the modular **CSPDarknet Backbone**, **PANet Neck with Attention Hooks**, and **Multi-Scale Detection Head**.
   - Implemented the full **Controlled Component Ablation Matrix** (M1: Baseline YOLOv5s, M2: Baseline + CLAHE, M3: Baseline + Attention, M4: Proposed Integrated Architecture).
   - Formulated the multi-task loss with **CIoU (Complete IoU)** bounding box regression, Objectness BCE, and Multi-class defect classification BCE.
   - Built the complete evaluation harness computing Precision, Recall, F1-score, and **mAP@0.5**.
4. **Tooling & Engineering Standards:** Configured reproducible package management via `uv` (`pyproject.toml`, `uv.lock`, `.python-version`) and automated workflows via `Makefile`.

---

## 2. Dataset Collection & Exploratory Data Analysis (EDA)

### 2.1 Dataset Specification
* **Dataset Name:** NEU-DET (Northeastern University Surface Defect Database)
* **Image Count:** 1,800 images ($200 \times 200$ resolution, 3 color channels)
* **Dataset Partition:** 1,620 training images (90%, 3,743 bounding boxes) and 180 test images (10%, 446 bounding boxes; 4,189 total defect instances)
* **Number of Classes:** 6 balanced categories (300 images per class):
  1. `crazing` (fine networked cracks)
  2. `inclusion` (foreign material embedded in metal)
  3. `patches` (surface localized discoloration/oxidation)
  4. `pitted_surface` (cavities and small depressions)
  5. `rolled-in_scale` (oxide pressed into surface during rolling)
  6. `scratches` (linear surface abrasions)
* **Annotation Format:** Standard normalized YOLO format `[class_id, x_center, y_center, width, height]`.
* **Generalization Benchmark:** GC10-DET (3,570 images, 10 defect classes) staged for cross-dataset generalization evaluation in Phase 2.

### 2.2 Dataset Verification & Ingestion Script
The dataset pipeline is managed via `scripts/download_dataset.py` and `src/dataset/loader.py`, featuring automated integrity checks, caching, and custom PyTorch `DataLoader` collation for variable ground-truth bounding box counts.

---

## 3. Image Preprocessing & Enhancement Implementation

### 3.1 Motivation & Theoretical Formulation
In industrial metal inspection lines, metallic surfaces suffer from severe specular glare, oil-slick reflections, and non-uniform factory illumination. Faint defect patterns (e.g. subtle crazing lines and faint scratches) frequently suffer from receptive field contrast dilution in standard CNN backbones.

To address **Hypothesis H1 (Illumination & Contrast Enhancement)**:
1. **Bilateral Filtering ($I_{bf}$):**
   $$I_{bf}(p) = \frac{1}{W_p} \sum_{q \in S} I(q) \cdot G_{\sigma_s}(\|p - q\|) \cdot G_{\sigma_r}(|I(p) - I(q)|)$$
   Smooths high-frequency grain noise without blurring sharp defect boundaries, using spatial Gaussian $G_{\sigma_s}$ ($\sigma_s=50.0$) and range Gaussian $G_{\sigma_r}$ ($\sigma_r=50.0$).
2. **CLAHE (Contrast-Limited Adaptive Histogram Equalization):**
   Divides the image into $8 \times 8$ contextual tiles, applies histogram equalization, and clips contrast amplification at limit $2.0$ to prevent noise over-amplification. Applied specifically to the Luminance ($L^*$) channel in CIELAB color space to prevent chromatic artifacts.

### 3.2 Visual & Quantitative Results
* Script: `scripts/visualize_preprocessing.py`
* Generated Artifacts: `reports/figures/preprocessing_comparison.png`
* Intermediate stages verified: `raw` $\rightarrow$ `bilateral` $\rightarrow$ `clahe_only` $\rightarrow$ `enhanced` $\rightarrow$ `sobel_edge_map`.

---

## 4. 25% Proposed Methodology Implementation

### 4.1 Lightweight Attention Modules (`src/models/attention.py`)
To test **Hypothesis H2 (Multi-Scale Feature Attention)**:
* **ECA (Efficient Channel Attention):** Replaces parameter-heavy fully connected layers with an adaptive 1D convolution over channel dimension:
  $$k = \psi(C) = \left| \frac{\log_2(C)}{\gamma} + \frac{b}{\gamma} \right|_{odd}, \quad \gamma=2, b=1$$
  Captures direct cross-channel interactions with negligible computational overhead (< 200 parameters total).
* **Spatial Attention Module (SAM):** Pools channel statistics via average pooling and max pooling, concatenated into a 2-channel map followed by a $7 \times 7$ convolution and sigmoid gating.
* **ECASpatialAttention:** Fuses ECA and Spatial attention in a sequential residual block.

### 4.2 Modular Architecture & Controlled Ablation Matrix
We implemented the 4 experimental models specified in Task 3:
* **M1 (Baseline):** Standard YOLOv5s detector (CSPDarknet + PANet neck) operating on raw images.
* **M2 (Baseline + Preprocessing):** YOLOv5s detector coupled with CLAHE + Bilateral filtering pipeline (verifies H1).
* **M3 (Baseline + Attention):** YOLOv5s detector with ECA + Spatial attention inserted into the PANet neck feature pyramid (verifies H2).
* **M4 (Proposed Integrated Model):** Combined CLAHE/Bilateral preprocessing + ECA/Spatial multi-scale attention (verifies peak synergy).

### 4.3 Multi-Scale Loss, Evaluation Harness & Hardware Benchmark
* **CIoU Loss:** Incorporates bounding box overlap area, centroid euclidean distance, and aspect ratio consistency $v$:
  $$\mathcal{L}_{CIoU} = 1 - IoU + \frac{\rho^2(b, b^{gt})}{c^2} + \alpha v$$
* **Multi-Task Objective:** $\mathcal{L}_{total} = \lambda_{box} \mathcal{L}_{CIoU} + \lambda_{obj} \mathcal{L}_{obj} + \lambda_{cls} \mathcal{L}_{cls}$.
* **Evaluation Metrics:** Implemented in `src/evaluation/metrics.py` computing per-class Precision, Recall, F1, and **mAP@0.5**.
* **Latency & Throughput Benchmarking:** Implemented in `src/evaluation/benchmark.py` and `scripts/benchmark_fps.py`, measuring both model forward pass latency and true end-to-end pipeline latency (including CPU-based CLAHE and bilateral filtering):

| Model | Description | Parameters | Model Size | Preproc Latency | Model Latency | E2E Latency | E2E FPS | Real-Time ($\ge 30$ FPS) |
|---|---|---|---|---|---|---|---|---|
| **M1** | Baseline (Plain YOLOv5s) | 7,199,651 | 27.46 MB | 0.00 ms | 3.23 ms | 3.23 ms | **310.1** | **PASS** |
| **M2** | Baseline + CLAHE & Bilateral | 7,199,651 | 27.46 MB | 1.01 ms | 3.26 ms | 4.27 ms | **234.0** | **PASS** |
| **M3** | Baseline + ECA/Spatial Attention | 7,199,960 | 27.47 MB | 0.00 ms | 3.51 ms | 3.51 ms | **285.2** | **PASS** |
| **M4** | Proposed Integrated Model | 7,199,960 | 27.47 MB | 1.00 ms | 3.56 ms | 4.56 ms | **219.3** | **PASS** |

All models comfortably exceed the 30–50 FPS real-time manufacturing threshold, confirming **Hypothesis H3**.

---

## 5. Major Tasks / Modules Remaining (The Remaining 75%)

| Major Remaining Module | Expected Deliverables & Outputs |
|---|---|
| **Full Multi-Epoch Convergence Training** across M1, M2, M3, and M4 on GPU | Checkpoint weights (`checkpoints/M1_best.pt` through `M4_best.pt`) and loss convergence curves |
| **Hyperparameter & Anchor Optimization** (fine-tuned anchor configurations, cosine LR scheduling) | Tuned anchor configurations and training hyperparameter grid |
| **Grad-CAM Saliency Interpretability Module** (generating explainable heatmaps over detector feature maps for operator feedback) | Saliency overlay visualization script & qualitative interpretability figures |
| **Cross-Dataset Generalization on GC10-DET** (evaluating zero-shot / fine-tuned domain transfer to evaluate H4) | Comparative generalization table (NEU-DET $\rightarrow$ GC10-DET) |
| **Deployment Latency Optimization & ONNX / TensorRT Export** | Quantized FP16/INT8 export and edge benchmark report ($>50\text{ FPS}$) |
| **Interactive End-to-End Inspection Dashboard / Demonstration** (Streamlit / Gradio web UI) | Functional defect inspection web demo with image upload, bbox detection, and Grad-CAM |
| **Final Technical Documentation & Oral Presentation** | Complete technical project report and presentation slide deck |
