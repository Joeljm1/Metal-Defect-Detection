The next step involves understanding the training process and the loss function. I will analyze the loss calculation, training loop, and the script that launches the...
  To master this codebase without feeling overwhelmed, you should follow a 6-stage chronological roadmap.

  Do not start by reading the complex neural network files first. Follow the natural lifecycle of computer vision: Image Processing → Dataset & Box Math → Model
  Architecture → Loss & Training → Evaluation → Deployment.
  ──────
  ┌───────────────────────────┐   ┌─────────────────────────────────┐   ┌───────────────────────────────┐   ┌───────────────────────┐
  │                           │   │                                 │   │                               │   │                       │
  │ Stage 1: Image Processing │   │     Stage 2: Bounding Boxes     │   │ Stage 3: Network Architecture │   │                       │
  │                           ├──►│                                 ├──►│                               ├──►│ P4["Stage 4: Training │
  │      (NumPy + OpenCV)     │   │       (box_ops + Datasets)      │   │      (PyTorch nn.Module)      │   │                       │
  │                           │   │                                 │   │                               │   │                       │
  └───────────────────────────┘   └─────────────────────────────────┘   └───────────────┬───────────────┘   └───────────────────────┘
                                                                                        │
                                                                                        │
  ┌───────────────────────────┐   ┌─────────────────────────────────┐                   │                   ┌───────────────────────┐
  │                           │   │                                 │                   │                   │                       │
  │                           │   │                                 │                   │                   │          Loss         │
  │             P4            ├──►│       P5["Stage 5: Metrics      │                   └──────────────────►│                       │
  │                           │   │                                 │                                       │  (CIoU + Autograd)"]  │
  │                           │   │                                 │                                       │                       │
  └─────────────┬─────────────┘   └─────────────────────────────────┘                                       └───────────────────────┘
                │
                └───────────────┐
  ┌───────────────────────────┐ │ ┌─────────────────────────────────┐
  │                           │ │ │                                 │
  │                           │ │ │            Evaluation           │
  │             P5            │ └►│                                 │
  │                           │   │        (mAP@0.5 + FPS)"]        │
  │                           │   │                                 │
  └─────────────┬─────────────┘   └─────────────────────────────────┘
                │
                │
                │                 ┌─────────────────────────────────┐
                │                 │                                 │
                ├────────────────►│         P6["Stage 6: XAI        │
                │                 │                                 │
                │                 └─────────────────────────────────┘
                │
                │
                │                 ┌─────────────────────────────────┐
                │                 │                                 │
                │                 │         Edge Deployment         │
                └────────────────►│                                 │
                                  │ (Grad-CAM + ONNX + Streamlit)"] │
                                  │                                 │
                                  └─────────────────────────────────┘
  ──────
  ## المرحلة 1: Stage 1 — Image Processing & Filtering (The Input Pipeline)

  Before an image ever touches a neural network, it is modified using classical computer vision.

  ### 1. Theory to Understand

  • Images as Matrices: How a 2D grid represents grayscale intensity (0 - 255), and how a 3D matrix represents channels.
  • RGB vs. BGR: OpenCV reads images in BGR order by default, whereas PyTorch and Matplotlib expect RGB.
  • CIELAB Color Space: Why separating pure Lightness (L^) from color opponents (a^, b^*) allows illumination normalization without discoloring the metal.
  • Histogram Equalization & CLAHE: Why standard equalization amplifies noise, and how local tiles with contrast clipping solve this.
  • Bilateral Filtering: Why Gaussian blur ruins sharp crack edges, and how weighting by both spatial distance and color difference preserves edges while smoothing grain.

  ### 2. Libraries to Learn (Only What’s Needed)

  • numpy: Array indexing (img[:, :, 0]), .shape, np.clip(), np.transpose().
  • opencv-python (cv2): cv2.imread(), cv2.cvtColor(), cv2.createCLAHE(), cv2.bilateralFilter().

  ### 3. Repository Files to Read (In Order)

  1. preprocessing.yaml — Look at the hyperparameters (clip_limit: 2.0, tile_grid: [8, 8], d: 5, sigma_color: 50.0).
  2. clahe.py — See how RGB is converted to LAB, equalized on the L-channel, and converted back.
  3. bilateral.py — See how bilateral smoothing is applied.
  4. pipeline.py — See how CLAHE and Bilateral are chained into a single DefectPreprocessor class.

  ### 4. Hands-on Command to Run

    make visualize

  Open reports/figures/preprocessing_comparison.png to visually see what your code just did to the raw steel images.
  ──────
  ## 📦 Stage 2: Bounding Boxes, Datasets & Anchors (Data Plumbing)

  Now that you know how images are cleaned, you need to understand how the computer knows where defects are located.

  ### 1. Theory to Understand

  • Bounding Box Formats:
      • Pascal VOC: [x_{min}, y_{min}, x_{max}, y_{max}] (in absolute pixel values).
      • YOLO format: [x_{center}, y_{center}, width, height] (normalized from 0.0 to 1.0).
  • Intersection over Union (IoU): Overlap Area/Union Area.
  • Anchor Boxes & K-Means Clustering: Why we cluster dataset bounding box shapes into 9 reference anchor boxes instead of guessing box sizes at random.

  ### 2. Libraries to Learn

  • xml.etree.ElementTree: Python's built-in tool for parsing Pascal VOC .xml annotation tags (<bndbox>, <xmin>, <ymin>).
  • torch.utils.data.Dataset & DataLoader: Understanding __len__(), __getitem__(), and why object detection requires a custom collate_fn (because different images
  contain different numbers of defect boxes).

  ### 3. Repository Files to Read (In Order)

  1. box_ops.py — Read functions xywh2xyxy(), xyxy2xywh(), and box_iou().
  2. parser.py — See how VOC XML annotations are parsed and normalized.
  3. loader.py — Read MetalDefectDataset and yolo_collate_fn.
  4. cluster_anchors.py — See how K-means clusters 9 anchor box templates.

  ### 4. Hands-on Command to Run

    make eda

  Inspect reports/dataset_stats.json to see defect aspect ratios and frequencies.
  ──────
  ## 🧠 Stage 3: Neural Network Architecture (The Model Anatomy)

  This is the core of the project. Break it down into modular parts rather than trying to digest the whole network at once.

  ### 1. Theory to Understand

  • PyTorch nn.Module: How __init__() defines layers and forward() defines the flow of tensors.
  • Basic Blocks: Convolutions, Batch Normalization, and SiLU activations (ConvBNSiLU).
  • CSP (Cross-Stage Partial) Connections: Why splitting the flow into a heavy path and a bypass highway reduces computation by ≈ 50%.
  • Attention Mechanisms:
      • ECA (Channel Attention): 1D convolution across channels to boost defect feature channels.
      • Spatial Attention: 7 × 7 convolution creating a 2D map to highlight defect locations.
  • Neck (FPN + PANet): Top-down and bottom-up multi-scale feature fusion.

  ### 2. Libraries to Learn

  • torch.nn: nn.Conv2d, nn.BatchNorm2d, nn.SiLU, nn.AdaptiveAvgPool2d, nn.Conv1d, nn.Sigmoid, nn.Sequential.

  ### 3. Repository Files to Read (In Order)

  1. attention.py — Start here! It is the shortest, cleanest file. Read ECABlock, SpatialAttentionBlock, and ECASpatialAttention.
  2. backbone.py — Read ConvBNSiLU, Bottleneck, C3Block, SPPFBlock, and then CSPDarknetBackbone.
  3. neck.py — Trace how features from the backbone are blended and where attention is inserted for M3/M4.
  4. head.py — Read how predictions are decoded into bounding boxes across P3 (80 × 80), P4 (40 × 40), and P5 (20 × 20).
  5. detector.py — Read DefectDetector to see how M1, M2, M3, and M4 are cleanly toggled with flags (use_preprocessing, use_attention).

  ### 4. Hands-on Command to Run

    uv run pytest tests/test_attention.py tests/test_models.py -v

  All tests will run and confirm that tensors flow through these layers with correct dimensions.
  ──────
  ## 🎯 Stage 4: Loss Functions & Training Loops

  How does the model actually learn from its mistakes?

  ### 1. Theory to Understand

  • Multi-Task Loss:

    ℒ      = λ   ℒ     + λ   ℒ    + λ   ℒ
     total    box CIoU    obj obj    cls cls

  • CIoU Loss: Why we penalize overlap, distance between box centers, and aspect ratio consistency.
  • Binary Cross Entropy with Logits (BCEWithLogitsLoss): Numerically stable classification loss.
  • Optimization: Learning rates, weight decay, AdamW/SGD, and Cosine Annealing schedulers.

  ### 2. Libraries to Learn

  • torch.optim: torch.optim.AdamW, lr_scheduler.CosineAnnealingLR.
  • PyTorch Autograd: loss.backward(), optimizer.step(), optimizer.zero_grad().

  ### 3. Repository Files to Read (In Order)

  1. loss.py — Read ComputeLoss.forward(). Notice how targets are matched to anchor grids and how CIoU, objectness, and class loss are calculated.
  2. trainer.py — Read train_epoch(), validate(), and checkpoint saving.
  3. train.py — The CLI script that connects configurations, dataloaders, models, and trainers.

  ### 4. Hands-on Command to Run

    make dry-run

  This executes a fast 1-epoch sanity run to verify the training and validation loops in action.
  ──────
  ## 📊 Stage 5: Evaluation, Metrics & Benchmarks

  How do we prove our model works and meets industrial criteria?

  ### 1. Theory to Understand

  • Precision, Recall, and F1-Score: Precision (avoiding false alarms) vs. Recall (not missing real defects).
  • Mean Average Precision (mAP@0.5): 101-point VOC interpolation calculating the area under the PR curve.
  • Inference Latency & FPS: Measuring milliseconds per frame to ensure ≥ 30 - 50 FPS.

  ### 2. Repository Files to Read (In Order)

  1. metrics.py — See how compute_ap(), compute_map(), and confidence thresholds are handled.
  2. benchmark.py — See how PyTorch GPU/CPU inference speed is timed over 100 warm-up and test runs.
  3. benchmark_fps.py — The standalone benchmarking runner.

  ### 3. Hands-on Command to Run

    make benchmark

  Outputs latency in milliseconds and real-time FPS on your hardware.
  ──────
  ## 🚀 Stage 6: Advanced Modules — Explainability, Transfer & Deployment

  The final phase bridges research into an industrial deployment product.

  ### 1. Theory to Understand

  • Explainable AI (Grad-CAM): Hooking into intermediate feature maps and using gradients to draw a saliency heat map.
  • Domain Adaptation: Testing a model trained on one dataset (NEU-DET) on an unseen dataset (GC10-DET) to test real-world transferability.
  • ONNX (Open Neural Network Exchange): Converting Python PyTorch models into a static, cross-platform computational graph for high-speed edge inference on CPUs without
  PyTorch.
  • Streamlit: Building interactive web tools in pure Python.

  ### 2. Repository Files to Read (In Order)

  1. gradcam.py — See how forward/backward hooks are registered on neck layers (c3_fpn2, c3_pan1, att_p3).
  2. domain_adaptation.py — See how the head is adapted for GC10-DET's 10 classes.
  3. export.py — See torch.onnx.export() and validation against numerical drift.
  4. dashboard.py & app.py — The interactive Streamlit user interface.

  ### 3. Hands-on Commands to Run

    # 1. Generate Grad-CAM heatmaps
    uv run python scripts/visualize_gradcam.py
    
    # 2. Launch the interactive web dashboard
    uv run streamlit run app.py
  ──────
  ## 📋 Recommended Weekly Learning Plan

   Milestone     | Focus Areas                                             | What You Will Understand
  ---------------|---------------------------------------------------------|---------------------------------------------------------------------------------------------
   Day 1–2       | Stage 1 (NumPy, OpenCV, CLAHE, Bilateral)               | How digital images work, why metal glare is difficult, and how filters clean images.
   Day 3–4       | Stage 2 (Box math, VOC XML, PyTorch DataLoaders)        | How bounding boxes are formatted, how IoU works, and how data feeds into training.
   Day 5–6       | Stage 3 (Backbone, PANet Neck, ECA & Spatial Attention) | How convolutional layers extract features, how bypass highways work, and what attention is.
   Day 7–8       | Stage 4 & 5 (Losses, CIoU, Training loop, )             | How backpropagation guides the model dials, and how accuracy and FPS are measured.
   Day 9–10      | Stage 6 (Grad-CAM, ONNX, Streamlit Dashboard)           | How the model is explained to factory operators and packaged for real-world deployment.
