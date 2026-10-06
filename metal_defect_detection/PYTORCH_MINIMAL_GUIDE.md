# 🎯 Minimal PyTorch Guide: Strictly What You Need for This Project

> **Goal:** Do not waste weeks studying NLP, audio, transformers, or distributed training.  
> This guide covers **only the exact ~15% of PyTorch** used in this metal defect detection codebase. You can master this in an afternoon.

---

## 📋 Table of Contents
1. [Cheat Sheet: What This Project Uses vs. What to Ignore](#1-cheat-sheet-what-this-project-uses-vs-what-to-ignore)
2. [Concept 1: Tensors (The Numbers & Shapes)](#2-concept-1-tensors-the-numbers--shapes)
3. [Concept 2: Tensor Reshaping & Permutation (`permute`, `view`, `unsqueeze`)](#3-concept-2-tensor-reshaping--permutation)
4. [Concept 3: Building Layers with `torch.nn`](#4-concept-3-building-layers-with-torchnn)
5. [Concept 4: The `forward()` Pass & Multi-Scale Feature Blending](#5-concept-4-the-forward-pass--multi-scale-feature-blending)
6. [Concept 5: The Custom `Dataset` & `DataLoader` (`collate_fn`)](#6-concept-5-the-custom-dataset--dataloader-collate_fn)
7. [Concept 6: The Golden 5-Line Training Loop](#7-concept-6-the-golden-5-line-training-loop)
8. [Concept 7: Saving, Loading & Inference (`torch.no_grad`)](#8-concept-7-saving-loading--inference-torchno_grad)
9. [Concept 8: Advanced Hooks for Grad-CAM (Bonus)](#9-concept-8-advanced-hooks-for-grad-cam-bonus)
10. [Where Each Concept Lives in This Repository](#10-where-each-concept-lives-in-this-repository)
11. [Top Free Resources (Ranked by Efficiency)](#11-top-free-resources-ranked-by-efficiency)

---

## 1. Cheat Sheet: What This Project Uses vs. What to Ignore

| ✅ MUST LEARN (Used in this Project) | ❌ IGNORE COMPLETELY (Not used here) |
| :--- | :--- |
| `torch.Tensor`, `.shape`, `.to(device)` | Recurrent Neural Networks (RNN / LSTM) |
| `.permute()`, `.unsqueeze()`, `.squeeze()` | Transformers, Self-Attention, MultiheadAttention |
| `nn.Module`, `nn.Conv2d`, `nn.BatchNorm2d`, `nn.SiLU` | Natural Language Processing (Tokenizers, Embeddings) |
| `nn.AdaptiveAvgPool2d`, `nn.Conv1d` (ECA Attention) | Audio processing (`torchaudio`) |
| `F.interpolate()` (FPN Upsampling) | Reinforcement Learning |
| `torch.utils.data.Dataset`, `DataLoader`, `collate_fn` | TorchText, Hugging Face Pipelines |
| `loss.backward()`, `optimizer.step()`, `zero_grad()` | Distributed Data Parallel (DDP / multi-node clusters) |
| `model.eval()`, `torch.no_grad()`, `torch.save/load` | TorchScript / JIT Tracing (we use ONNX instead) |

---

## 2. Concept 1: Tensors (The Numbers & Shapes)

A **Tensor** is PyTorch's version of a NumPy array, with two superpowers:
1. It tracks math gradients automatically for learning (`requires_grad=True`).
2. It can run on a GPU (`.to('cuda')`) for $50\times$ faster matrix operations.

### The Universal Image Tensor Shape: `[B, C, H, W]`
In computer vision, every batch of images is represented as a 4D Tensor:
```python
import torch

# Batch of 16 images, 3 color channels (RGB), 200 pixels tall, 200 pixels wide
x = torch.zeros(16, 3, 200, 200)

print(x.shape)   # torch.Size([16, 3, 200, 200])
print(x.dtype)   # torch.float32 (standard for neural networks)
print(x.device)  # cpu (or cuda:0)
```

### Essential Device Switching (CPU $\leftrightarrow$ GPU)
In this project, code automatically detects if you have an NVIDIA GPU:
```python
device = "cuda" if torch.cuda.is_available() else "cpu"

# Move model and data to the chosen device
x = x.to(device)
```

---

## 3. Concept 2: Tensor Reshaping & Permutation

This is where beginners get stuck. Computer vision frequently transforms images between:
- OpenCV / Matplotlib format: `(Height, Width, Channels)`
- PyTorch format: `(Channels, Height, Width)` or `(Batch, Channels, Height, Width)`

Here are the only 4 reshaping functions you need:

### 1. `permute(*dims)` — Swapping Dimensions
Used in [`src/models/detector.py`](file:///home/joeljm/Downloads/clg/sem%207/computer_vision/proj/metal_defect_detection/src/models/detector.py#L204):
```python
# NumPy image from OpenCV: Height=200, Width=200, Channels=3
img_np = np.zeros((200, 200, 3), dtype=np.uint8)

# Convert to PyTorch: (200, 200, 3) -> (3, 200, 200)
tensor_chw = torch.from_numpy(img_np).permute(2, 0, 1)
```

### 2. `unsqueeze(dim)` & `squeeze(dim)` — Adding/Removing Fake Dimensions
Used to turn a single image into a "batch of 1":
```python
# Single image: shape is [3, 200, 200]
img = torch.zeros(3, 200, 200)

# Add batch dimension at index 0: shape becomes [1, 3, 200, 200]
batch = img.unsqueeze(0)

# Remove batch dimension: shape goes back to [3, 200, 200]
single = batch.squeeze(0)
```

### 3. `view()` or `reshape()` — Flattening / Unrolling Grids
Used in [`src/models/head.py`](file:///home/joeljm/Downloads/clg/sem%207/computer_vision/proj/metal_defect_detection/src/models/head.py) to reshape raw feature maps into anchor box predictions:
```python
# Shape: [Batch=16, Anchors=3, Boxes=4, H=20, W=20]
raw_pred = torch.zeros(16, 3, 4, 20, 20)

# Flatten spatial grid into total predictions: [16, 3 * 20 * 20, 4] -> [16, 1200, 4]
flattened = raw_pred.view(16, -1, 4)
```

---

## 4. Concept 3: Building Layers with `torch.nn`

Every model in PyTorch inherits from `nn.Module`. It always follows this pattern:
1. `__init__()`: Declare your layers.
2. `forward(x)`: Define how data flows through them.

### Real Example from This Codebase: `ConvBNSiLU`
Open [`src/models/backbone.py`](file:///home/joeljm/Downloads/clg/sem%207/computer_vision/proj/metal_defect_detection/src/models/backbone.py#L20-L40). You will see the standard convolutional brick used everywhere:

```python
import torch.nn as nn

class ConvBNSiLU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        # 1. 2D Convolution (sliding window filter)
        self.conv = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=kernel_size // 2,  # keeps width & height the same if stride=1
            bias=False
        )
        # 2. Batch Normalization (stabilizes training numbers)
        self.bn = nn.BatchNorm2d(out_channels)
        # 3. Activation Function (decision gate: SiLU / Swish)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input -> Conv -> BatchNorm -> SiLU -> Output
        return self.act(self.bn(self.conv(x)))
```

### The 4 Specialized Layers Used in Our Attention Modules
Look at [`src/models/attention.py`](file:///home/joeljm/Downloads/clg/sem%207/computer_vision/proj/metal_defect_detection/src/models/attention.py):
* `nn.AdaptiveAvgPool2d(1)`: Condenses an entire $(H \times W)$ image into a single average value per channel.
* `nn.Conv1d(1, 1, kernel_size=k)`: 1D convolution across channel vectors for **ECA attention**.
* `nn.Sigmoid()`: Squashes numbers into a $[0.0, 1.0]$ attention mask.
* `nn.Identity()`: A dummy do-nothing layer used when an ablation model disables attention.

---

## 5. Concept 4: The `forward()` Pass & Multi-Scale Feature Blending

In YOLO and PANet, we combine features from different network depths:
* **Downsampling:** Done using `Conv2d` with `stride=2` (cuts resolution in half: $80 \to 40 \to 20$).
* **Upsampling:** Done using `F.interpolate()` (doubles resolution: $20 \to 40 \to 80$).
* **Concatenation:** Done using `torch.cat([tensor_a, tensor_b], dim=1)` (glues feature channels together).

### How PANet Blends Features in [`src/models/neck.py`](file:///home/joeljm/Downloads/clg/sem%207/computer_vision/proj/metal_defect_detection/src/models/neck.py):
```python
import torch
import torch.nn.functional as F

# P5 is deep and small: [Batch=16, Channels=512, Height=20, Width=20]
# P4 is medium:        [Batch=16, Channels=256, Height=40, Width=40]

# 1. Upsample P5 to match P4's height and width (20x20 -> 40x40)
p5_upsampled = F.interpolate(p5, size=p4.shape[2:], mode="nearest")

# 2. Concatenate along channel dimension (dim=1)
# 256 channels + 256 channels = 512 channels
fused = torch.cat([p4, p5_upsampled], dim=1)
```

---

## 6. Concept 5: The Custom `Dataset` & `DataLoader`

PyTorch needs a way to feed images from disk into GPU memory.

1. **`Dataset`**: Defines how to load **one single image and its boxes**.
2. **`DataLoader`**: Shuffles data, groups it into batches of 16, and feeds it across CPU threads.

### Why Object Detection Needs a Custom `collate_fn`
In simple classification (cats vs. dogs), every image has exactly 1 label.  
In object detection:
- Image 1 might have **3 scratches**.
- Image 2 might have **12 pits**.
- Image 3 might have **0 defects**.

Standard PyTorch fails to stack arrays of different lengths! Our custom `yolo_collate_fn` in [`src/dataset/loader.py`](file:///home/joeljm/Downloads/clg/sem%207/computer_vision/proj/metal_defect_detection/src/dataset/loader.py#L90-L115) prepends the batch index to each box:
```python
# Output targets tensor: [batch_index, class_id, x_center, y_center, width, height]
# Row 0: [0, 5, 0.45, 0.22, 0.10, 0.80] -> In Image 0, there is a Scratch (class 5)
# Row 1: [0, 5, 0.52, 0.31, 0.08, 0.75] -> In Image 0, there is a 2nd Scratch
# Row 2: [1, 2, 0.10, 0.80, 0.30, 0.25] -> In Image 1, there is a Patch (class 2)
```

---

## 7. Concept 6: The Golden 5-Line Training Loop

Every deep learning training loop in PyTorch boils down to these exact 5 lines. Look at [`src/training/trainer.py`](file:///home/joeljm/Downloads/clg/sem%207/computer_vision/proj/metal_defect_detection/src/training/trainer.py#L65-L80):

```python
# 1. Clear leftover gradients from previous step
optimizer.zero_grad()

# 2. Forward pass: Model makes its predictions
predictions = model(images)

# 3. Calculate penalty score using CIoU + BCE loss
loss, loss_items = compute_loss(predictions, targets)

# 4. Backward pass: Calculate gradients (chain rule)
loss.backward()

# 5. Step optimizer: Nudge network dials in opposite direction of error
optimizer.step()
```

---

## 8. Concept 7: Saving, Loading & Inference

### 1. Disabling Gradient Tracking for Evaluation / Inference
When inspecting metal sheets, we do **not** want PyTorch to remember gradients (saving RAM and speeding up execution):
```python
model.eval()  # Put layers like BatchNorm in evaluation mode

with torch.no_grad():  # Turn off calculus engine
    detections = model(test_image)
```

### 2. Saving and Loading Model Checkpoints
In [`src/training/trainer.py`](file:///home/joeljm/Downloads/clg/sem%207/computer_vision/proj/metal_defect_detection/src/training/trainer.py#L110):
```python
# Saving best weights
torch.save({
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "best_map": best_map,
}, "checkpoints/M4_best.pt")

# Loading weights back for testing
checkpoint = torch.load("checkpoints/M4_best.pt", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
```

---

## 9. Concept 8: Advanced Hooks for Grad-CAM (Bonus)

To generate explainable heat maps ([`src/evaluation/gradcam.py`](file:///home/joeljm/Downloads/clg/sem%207/computer_vision/proj/metal_defect_detection/src/evaluation/gradcam.py)), we intercept feature maps inside the network using **hooks**:

```python
# Save intermediate activations during forward pass
def forward_hook(module, input, output):
    activations.append(output)

# Attach hook to the neck's P3 attention layer
handle = model.neck.att_p3.register_forward_hook(forward_hook)
```

---

## 10. Where Each Concept Lives in This Repository

| PyTorch Concept | Exact File in This Project |
| :--- | :--- |
| **`permute`, `unsqueeze`, `no_grad`** | [`src/models/detector.py`](file:///home/joeljm/Downloads/clg/sem%207/computer_vision/proj/metal_defect_detection/src/models/detector.py#L181-L218) |
| **`nn.Conv2d`, `BatchNorm2d`, `SiLU`** | [`src/models/backbone.py`](file:///home/joeljm/Downloads/clg/sem%207/computer_vision/proj/metal_defect_detection/src/models/backbone.py#L20-L40) |
| **`F.interpolate`, `torch.cat`** | [`src/models/neck.py`](file:///home/joeljm/Downloads/clg/sem%207/computer_vision/proj/metal_defect_detection/src/models/neck.py#L60-L90) |
| **`nn.AdaptiveAvgPool2d`, `nn.Conv1d`** | [`src/models/attention.py`](file:///home/joeljm/Downloads/clg/sem%207/computer_vision/proj/metal_defect_detection/src/models/attention.py#L20-L55) |
| **`Dataset`, `collate_fn`, `DataLoader`** | [`src/dataset/loader.py`](file:///home/joeljm/Downloads/clg/sem%207/computer_vision/proj/metal_defect_detection/src/dataset/loader.py#L16-L115) |
| **`loss.backward()`, `optimizer.step()`** | [`src/training/trainer.py`](file:///home/joeljm/Downloads/clg/sem%207/computer_vision/proj/metal_defect_detection/src/training/trainer.py#L60-L85) |
| **`torch.clamp`, `torch.where`, CIoU math** | [`src/training/loss.py`](file:///home/joeljm/Downloads/clg/sem%207/computer_vision/proj/metal_defect_detection/src/training/loss.py#L25-L95) |
| **`register_forward_hook`** | [`src/evaluation/gradcam.py`](file:///home/joeljm/Downloads/clg/sem%207/computer_vision/proj/metal_defect_detection/src/evaluation/gradcam.py#L40-L65) |

---

## 11. Top Free Resources (Ranked by Efficiency)

### 🥇 1. Interactive & Quick (Spend 45 Minutes Here)
* **PyTorch Official 60-Minute Blitz:**  
  [https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)  
  *Read only sections: "Tensors", "A Gentle Introduction to `torch.autograd`", and "Neural Networks".*
* **Visualizing Tensors Interactively (BetterExplained):**  
  Focus on 4D tensor indexing `[Batch, Channel, Height, Width]`.

### 🥈 2. Video Walkthroughs (Watch on 1.5x Speed)
* **"PyTorch for Deep Learning in 1 Hour" (freeCodeCamp / Daniel Bourke):**  
  [YouTube Search: "Daniel Bourke PyTorch beginner"]  
  *Watch only the first 60 minutes covering Tensors, `nn.Module`, and the standard training loop.*
* **"Convolutions and Feature Maps Explained Visually" (3Blue1Brown):**  
  [YouTube: 3Blue1Brown - Convolutional Neural Networks]  
  *The cleanest 15-minute visual explanation of sliding kernels.*

### 🥉 3. The 3 Official Docs Pages to Bookmark
* **`torch.Tensor` operations:** [https://pytorch.org/docs/stable/tensors.html](https://pytorch.org/docs/stable/tensors.html)
* **`torch.nn.Conv2d` parameters:** [https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)
* **`torch.utils.data.DataLoader`:** [https://pytorch.org/docs/stable/data.html](https://pytorch.org/docs/stable/data.html)

---

### 💡 Suggested 2-Hour Study Plan
1. **Hour 1:** Read Sections 2, 3, 4, and 7 of this document while testing the small code blocks in a Python terminal (`uv run python`).
2. **Hour 2:** Open [`src/models/attention.py`](file:///home/joeljm/Downloads/clg/sem%207/computer_vision/proj/metal_defect_detection/src/models/attention.py) and [`src/models/backbone.py`](file:///home/joeljm/Downloads/clg/sem%207/computer_vision/proj/metal_defect_detection/src/models/backbone.py) side-by-side with Section 4 of this guide. Trace how tensors enter and exit each layer.
