# How to Build an Object Detection Model from Scratch in PyTorch
## A Technical Step-by-Step Guide to `02_inspect_baseline_model.py`

This guide explains every component of [`02_inspect_baseline_model.py`](file:///home/joeljm/Downloads/clg/sem%207/computer_vision/proj/metal_defect_detection/learn/02_inspect_baseline_model.py). 

There are no analogies here. Every explanation is grounded in **tensor shapes**, **matrix operations**, and **PyTorch mechanics** so you can write custom computer vision architectures yourself.

---

## 1. The High-Level Architecture Contract

Before writing any neural network, you must define its **input contract** and **output contract**:

```
Input:  Tensor of shape (B, 3, 200, 200)   [Batch, RGB Channels, Height, Width]
Output: Tensor of shape (B, 2529, 11)      [Batch, Candidate Boxes, Box Attributes]
```

Where each of the 11 attributes in a candidate box is:
- **`0:2`**: Center coordinates $(x_c, y_c)$ in pixels $[0, 200]$
- **`2:4`**: Box dimensions $(w, h)$ in pixels $[0, 200]$
- **`4:5`**: Objectness score $P(\text{defect}) \in [0, 1]$
- **`5:11`**: Probabilities for the 6 defect classes $[\text{cls}_0, \dots, \text{cls}_5] \in [0, 1]$

To get from `(B, 3, 200, 200)` to `(B, 2529, 11)`, the network executes three sequential stages:
1. **Backbone**: Downsamples spatial resolution while expanding feature channels (Feature Extraction).
2. **Neck**: Combines coarse semantic features with fine edge features (Multi-Scale Fusion).
3. **Head**: Convolves features into box coordinates and decodes grid offsets into pixel boxes (Prediction & Decoding).

---

## 2. Block 1: `ConvBNSiLU` (The Atomic Unit)

Every standard layer in YOLOv5 combines three operations: **Convolution $\to$ Batch Normalization $\to$ Activation**.

```python
def autopad(k: int, p: int | None = None) -> int:
    return k // 2 if p is None else p

class ConvBNSiLU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 1, stride: int = 1, padding: int | None = None):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding=autopad(kernel_size, padding),
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))
```

### Technical Details to Remember:
1. **`bias=False` in `nn.Conv2d`**: 
   When a convolution is immediately followed by `nn.BatchNorm2d`, the convolution bias is mathematically redundant.
   $$\text{BN}(W x + b) = \gamma \left( \frac{(W x + b) - \mu}{\sigma} \right) + \beta = \gamma \left( \frac{W x - \mu_{Wx}}{\sigma} \right) + \beta$$
   The constant $b$ cancels out during mean subtraction. Setting `bias=False` saves memory and prevents gradient waste.
2. **`autopad(k)`**:
   For any odd kernel size $k$ with `stride=1`, setting `padding = k // 2` (e.g., padding=1 for $3 \times 3$, padding=3 for $7 \times 7$) ensures the output spatial dimensions $(H, W)$ exactly match the input.
3. **Why SiLU ($x \cdot \sigma(x)$) over ReLU**:
   ReLU zeroes out all negative values ($\frac{d}{dx}=0$ for $x < 0$), which can cause gradient death in deep networks. SiLU is smooth, continuously differentiable, and maintains non-zero gradients for small negative inputs.

---

## 3. Block 2: `Bottleneck` (Residual Shortcut)

```python
class Bottleneck(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, shortcut: bool = True, expansion: float = 0.5):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.cv1 = ConvBNSiLU(in_channels, hidden_channels, kernel_size=1, stride=1)
        self.cv2 = ConvBNSiLU(hidden_channels, out_channels, kernel_size=3, stride=1)
        self.add = shortcut and (in_channels == out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.cv2(self.cv1(x))
        return x + out if self.add else out
```

### Technical Details to Remember:
1. **Channel Compression (`expansion=0.5`)**:
   Instead of running a heavy $3 \times 3$ convolution across all channels, `cv1` ($1 \times 1$) reduces channels by 50%. The $3 \times 3$ convolution (`cv2`) operates on this compressed channel space and expands back. This cuts floating-point operations (FLOPs) by nearly half.
2. **Identity Addition (`x + out`)**:
   Residual addition provides an uninterrupted gradient backpropagation path: $\frac{\partial}{\partial x}(x + F(x)) = 1 + \frac{\partial F}{\partial x}$. Even if $\frac{\partial F}{\partial x}$ vanishes, a unit gradient propagates back.
3. **The Dimension Check**:
   You can only add tensors element-wise if their shapes match: `self.add = shortcut and (in_channels == out_channels)`.

---

## 4. Block 3: `C3Block` (Cross-Stage Partial Network)

The C3 block is the core building block of CSPDarknet.

```python
class C3Block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n: int = 1, shortcut: bool = True):
        super().__init__()
        hidden_channels = int(out_channels * 0.5)
        self.cv1 = ConvBNSiLU(in_channels, hidden_channels, kernel_size=1, stride=1)
        self.cv2 = ConvBNSiLU(in_channels, hidden_channels, kernel_size=1, stride=1)
        self.cv3 = ConvBNSiLU(2 * hidden_channels, out_channels, kernel_size=1, stride=1)
        self.m = nn.Sequential(
            *(Bottleneck(hidden_channels, hidden_channels, shortcut=shortcut) for _ in range(n))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch_a = self.m(self.cv1(x))
        branch_b = self.cv2(x)
        return self.cv3(torch.cat((branch_a, branch_b), dim=1))
```

### Mathematical Dataflow in C3:
```
                Input Tensor: (B, C_in, H, W)
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
     cv1: (B, C_h, H, W)         cv2: (B, C_h, H, W)
             │                   (Bypass Path)
     n x Bottlenecks                     │
             │                           │
     branch_a: (B, C_h, H, W)            │
             │                           │
             └─────────────┬─────────────┘
                           ▼
          torch.cat: (B, 2*C_h, H, W)
                           │
             cv3: (B, C_out, H, W)
```
- **Why CSP works**: Standard ResNets process 100% of channels through bottleneck convolutions, generating redundant gradient updates. CSP splits channels into two halves: branch A processes deep non-linearities, while branch B bypasses the computation entirely. Concatenating them at the end preserves representational capacity while cutting computation by ~50%.

---

## 5. Block 4: `SPPF` (Spatial Pyramid Pooling Fast)

SPPF sits at the deepest layer of the backbone (after stride 32).

```python
class SPPF(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, k: int = 5):
        super().__init__()
        hidden_channels = in_channels // 2
        self.cv1 = ConvBNSiLU(in_channels, hidden_channels, kernel_size=1, stride=1)
        self.cv2 = ConvBNSiLU(hidden_channels * 4, out_channels, kernel_size=1, stride=1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.cv2(torch.cat((x, y1, y2, y3), dim=1))
```

### Why Serial $5 \times 5$ Pooling Replaces Parallel $5 \times 5, 9 \times 9, 13 \times 13$:
- In spatial pooling: two consecutive $5 \times 5$ max-poolings with stride 1 have an effective receptive field of $9 \times 9$. Three consecutive $5 \times 5$ max-poolings have an effective receptive field of $13 \times 13$.
- By executing $5 \times 5$ serially (`y1 = m(x)`, `y2 = m(y1)`, `y3 = m(y2)`), SPPF reuses intermediate pooling buffers, achieving identical receptive fields at more than **$2\times$ the speed** of computing large kernels independently.
- Four tensors of shape `(B, hidden, 7, 7)` are concatenated into `(B, 4*hidden, 7, 7)` and compressed back by `cv2`.

---

## 6. Block 5: `CSPDarknetBackbone` (Feature Extraction)

The backbone downsamples an image from $(200, 200)$ to three hierarchical feature maps:

```python
class CSPDarknetBackbone(nn.Module):
    def __init__(self, in_channels: int = 3, depth_multiple: float = 0.33, width_multiple: float = 0.50):
        super().__init__()
        # Width scaling for YOLOv5s: 64*0.5=32, 128*0.5=64, 256*0.5=128, 512*0.5=256, 1024*0.5=512
        c1, c2, c3, c4, c5 = 32, 64, 128, 256, 512
        # Depth scaling: number of bottlenecks per C3 block
        n1, n2, n3 = 1, 2, 3

        self.stem = ConvBNSiLU(in_channels, c1, kernel_size=6, stride=2, padding=2)  # Stride 2
        self.stage1 = nn.Sequential(ConvBNSiLU(c1, c2, 3, 2), C3Block(c2, c2, n=n1)) # Stride 4
        self.stage2 = nn.Sequential(ConvBNSiLU(c2, c3, 3, 2), C3Block(c3, c3, n=n2)) # Stride 8  -> P3
        self.stage3 = nn.Sequential(ConvBNSiLU(c3, c4, 3, 2), C3Block(c4, c4, n=n3)) # Stride 16 -> P4
        self.stage4 = nn.Sequential(ConvBNSiLU(c4, c5, 3, 2), C3Block(c5, c5, n=n1), SPPF(c5, c5, 5)) # Stride 32 -> P5

        self.out_channels = [c3, c4, c5]  # [128, 256, 512]

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        x = self.stage1(x)
        p3 = self.stage2(x)
        p4 = self.stage3(p3)
        p5 = self.stage4(p4)
        return p3, p4, p5
```

### Exact Tensor Tracing Through the Backbone:
```
Input:   (1,   3, 200, 200)
Stem:    (1,  32, 100, 100)  [Stride 2]
Stage 1: (1,  64,  50,  50)  [Stride 4]
Stage 2: (1, 128,  25,  25)  [Stride 8]  ==> Output P3 (Fine spatial features)
Stage 3: (1, 256,  13,  13)  [Stride 16] ==> Output P4 (Medium-scale features)
Stage 4: (1, 512,   7,   7)  [Stride 32] ==> Output P5 (Coarse semantic features)
```

---

## 7. Block 6: `PANetNeck` (Multi-Scale Feature Fusion)

The neck fuses multi-scale features in two passes: **Top-Down (FPN)** and **Bottom-Up (PAN)**.

```python
class PANetNeck(nn.Module):
    def __init__(self, in_channels: list[int] = [128, 256, 512]):
        super().__init__()
        c3, c4, c5 = in_channels

        # Top-Down FPN Pathway
        self.reduce_p5 = ConvBNSiLU(c5, c4, kernel_size=1, stride=1)
        self.c3_fpn1 = C3Block(c4 + c4, c4, n=1, shortcut=False)

        self.reduce_p4 = ConvBNSiLU(c4, c3, kernel_size=1, stride=1)
        self.c3_fpn2 = C3Block(c3 + c3, c3, n=1, shortcut=False)

        # Bottom-Up PANet Pathway
        self.down_p3 = ConvBNSiLU(c3, c3, kernel_size=3, stride=2)
        self.c3_pan1 = C3Block(c3 + c4, c4, n=1, shortcut=False)

        self.down_p4 = ConvBNSiLU(c4, c4, kernel_size=3, stride=2)
        self.c3_pan2 = C3Block(c4 + c5, c5, n=1, shortcut=False)

    def forward(self, features: tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        p3, p4, p5 = features

        # 1. Top-Down FPN (Inject semantic context into fine layers)
        p5_reduced = self.reduce_p5(p5)                                         # (B, 256, 7, 7)
        p5_up = F.interpolate(p5_reduced, size=p4.shape[2:], mode="nearest")    # (B, 256, 13, 13)
        p4_fused = self.c3_fpn1(torch.cat([p5_up, p4], dim=1))                  # (B, 256, 13, 13)

        p4_reduced = self.reduce_p4(p4_fused)                                   # (B, 128, 13, 13)
        p4_up = F.interpolate(p4_reduced, size=p3.shape[2:], mode="nearest")    # (B, 128, 25, 25)
        p3_out = self.c3_fpn2(torch.cat([p4_up, p3], dim=1))                   # (B, 128, 25, 25)

        # 2. Bottom-Up PANet (Inject localization boundaries into coarse layers)
        p3_down = self.down_p3(p3_out)                                          # (B, 128, 13, 13)
        p4_out = self.c3_pan1(torch.cat([p3_down, p4_fused], dim=1))            # (B, 256, 13, 13)

        p4_down = self.down_p4(p4_out)                                          # (B, 256, 7, 7)
        p5_out = self.c3_pan2(torch.cat([p4_down, p5], dim=1))                 # (B, 512, 7, 7)

        return p3_out, p4_out, p5_out
```

### Why Two Separate Pathways?
- **$P_3$ ($25 \times 25$)**: Knows **WHERE** pixels and boundaries are, but has weak semantic understanding of the defect category.
- **$P_5$ ($7 \times 7$)**: Knows **WHAT** the defect category is, but has lost spatial resolution due to repeated downsampling.
- **Top-Down (FPN)** transfers "WHAT" down into $P_3$.
- **Bottom-Up (PAN)** transfers "WHERE" up into $P_5$.
- Result: Every pyramid level now possesses both spatial precision and semantic category awareness.

---

## 8. Block 7: `DetectHead` & Bounding Box Decoding

The head consists of three independent $1 \times 1$ convolutions that map feature channels into dense anchor predictions.

```python
class DetectHead(nn.Module):
    def __init__(self, num_classes: int = 6, in_channels: list[int] = [128, 256, 512], img_size=(200, 200)):
        super().__init__()
        self.num_classes = num_classes
        self.no = num_classes + 5  # 11 values: 4 bbox coords + 1 obj + 6 class logits
        self.img_size = img_size

        # 3 anchor aspect ratios per detection scale (derived via k-means on training dataset)
        raw_anchors = [
            [[29, 38], [34, 76], [79, 56]],       # P3 / stride 8 (small defects)
            [[33, 185], [57, 122], [178, 51]],    # P4 / stride 16 (medium defects)
            [[127, 80], [95, 182], [179, 186]],  # P5 / stride 32 (large defects)
        ]
        self.na = 3  # 3 anchors per grid cell
        self.register_buffer("anchors", torch.tensor(raw_anchors, dtype=torch.float32))

        # 1x1 convolutions producing predictions for each scale
        self.heads = nn.ModuleList([
            nn.Conv2d(c, self.na * self.no, kernel_size=1) for c in in_channels
        ])

        # Objectness bias initialization (-4.6 => sigmoid(-4.6) ~ 0.01)
        for module in self.heads:
            if isinstance(module, nn.Conv2d) and module.bias is not None:
                b = module.bias.view(self.na, -1)
                b.data[:, 4].fill_(-4.6)
                module.bias = nn.Parameter(b.view(-1), requires_grad=True)
```

### The YOLOv5 Box Decoding Math (Forward Pass)

The raw convolutional output contains unconstrained real numbers $(t_x, t_y, t_w, t_h, t_{\text{obj}}, t_{\text{cls}})$. They must be mapped into actual pixel coordinates:

```python
    def forward(self, features):
        raw_outputs, decoded_outputs = [], []
        for i, feat in enumerate(features):
            b, _, ny, nx = feat.shape
            
            # Reshape raw conv output: (B, 33, ny, nx) -> (B, 3, ny, nx, 11)
            out = self.heads[i](feat).view(b, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            raw_outputs.append(out)

            # 1. Grid coordinate matrix: (1, 1, ny, nx, 2)
            yv, xv = torch.meshgrid(torch.arange(ny, device=feat.device), torch.arange(nx, device=feat.device), indexing="ij")
            grid = torch.stack((xv, yv), dim=2).view(1, 1, ny, nx, 2).float()

            # 2. Anchor sizes for scale i: (1, 3, 1, 1, 2)
            anchor_grid = self.anchors[i].view(1, self.na, 1, 1, 2).to(feat.device)

            # 3. Grid cell stride in pixels: [200/nx, 200/ny]
            stride_t = feat.new_tensor([float(self.img_size[0]) / nx, float(self.img_size[1]) / ny]).view(1, 1, 1, 1, 2)

            # 4. Box Decoding Formulas
            xy = (out[..., 0:2].sigmoid() * 2.0 - 0.5 + grid) * stride_t
            wh = (out[..., 2:4].sigmoid() * 2.0) ** 2 * anchor_grid
            box = torch.cat((xy, wh), dim=-1)

            # 5. Objectness & Class Probabilities
            conf = out[..., 4:5].sigmoid()
            cls_prob = out[..., 5:].sigmoid()

            # 6. Flatten spatial dimensions: (B, na * ny * nx, 11)
            pred = torch.cat((box, conf, cls_prob), dim=-1).view(b, -1, self.no)
            decoded_outputs.append(pred)

        # Concatenate candidate boxes from all 3 scales: (B, 2529, 11)
        return torch.cat(decoded_outputs, dim=1), raw_outputs
```

### Derivation of the Decoding Formulas:

1. **Center Coordinates $(x_c, y_c)$**:
   $$x_c = \left( 2 \cdot \sigma(t_x) - 0.5 + c_x \right) \cdot \text{stride}_x$$
   - $\sigma(t_x) \in (0, 1)$.
   - Multiplying by 2 and subtracting 0.5 scales the offset to $(-0.5, 1.5)$.
   - This allows the predicted box center to move outside its current grid cell into neighboring cells, eliminating the grid sensitivity bottleneck.
   - Adding $c_x$ places it at grid cell index $c_x \in [0, N-1]$.
   - Multiplying by $\text{stride}_x$ converts grid indices into pixel coordinates $[0, 200]$.

2. **Dimensions $(w, h)$**:
   $$w = \left( 2 \cdot \sigma(t_w) \right)^2 \cdot \text{anchor}_w$$
   - Traditional YOLOv3 used exponential scaling: $w = a_w \cdot e^{t_w}$. However, $e^{t_w}$ is mathematically unbounded and causes gradient explosions if $t_w$ becomes large during early training.
   - YOLOv5 uses $\left( 2 \cdot \sigma(t_w) \right)^2$, which is strictly bounded in $(0, 4)$. The model can scale an anchor box between $0\times$ and $4\times$ its reference size, guaranteeing numerical stability.

3. **Total Candidate Box Count**:
   $$\text{Scale 1 } (P_3): 25 \times 25 \times 3 = 1,875\text{ boxes}$$
   $$\text{Scale 2 } (P_4): 13 \times 13 \times 3 = 507\text{ boxes}$$
   $$\text{Scale 3 } (P_5): 7 \times 7 \times 3 = 147\text{ boxes}$$
   $$\mathbf{\text{Total Boxes}} = 1,875 + 507 + 147 = \mathbf{2,529\text{ candidate boxes}}$$

---

## 9. Block 8: `BaselineDefectDetector` (Unified Module)

The entire detector combines the three components in 15 lines:

```python
class BaselineDefectDetector(nn.Module):
    def __init__(self, num_classes: int = 6, in_channels: int = 3, img_size: tuple[int, int] = (200, 200)):
        super().__init__()
        self.backbone = CSPDarknetBackbone(in_channels=in_channels, depth_multiple=0.33, width_multiple=0.50)
        self.neck = PANetNeck(in_channels=self.backbone.out_channels)
        self.head = DetectHead(num_classes=num_classes, in_channels=self.neck.out_channels, img_size=img_size)

    def forward(self, x: torch.Tensor):
        p3, p4, p5 = self.backbone(x)
        n3, n4, n5 = self.neck((p3, p4, p5))
        decoded, raw_outputs = self.head((n3, n4, n5))
        return decoded, raw_outputs
```

### Parameter Breakdown:
- **Backbone (CSPDarknet)**: 4,171,456 parameters (57.9%)
- **Neck (PANet)**: 2,998,528 parameters (41.6%)
- **Head (DetectHead)**: 29,667 parameters (0.4%)
- **Total Parameters**: **7,199,651 (~7.20 Million)**

---

## 10. Blueprint: How to Write Your Own Custom Detector

Whenever you need to write an object detection model in PyTorch from scratch, follow this checklist:

1. **Calculate Strides & Feature Map Resolutions**:
   For input resolution $W \times H$:
   - Stride 8: $\lceil W / 8 \rceil \times \lceil H / 8 \rceil$
   - Stride 16: $\lceil W / 16 \rceil \times \lceil H / 16 \rceil$
   - Stride 32: $\lceil W / 32 \rceil \times \lceil H / 32 \rceil$
2. **Channel Progression**:
   Standard scaling: $32 \to 64 \to 128 \to 256 \to 512$.
3. **Head Output Channels**:
   Formula: $\text{channels} = N_{\text{anchors}} \times (4 + 1 + N_{\text{classes}})$.
4. **Bias Initialization for Objectness**:
   Always initialize the objectness bias to $-4.6$. Because $>98\%$ of grid cells are background, a default zero bias causes the loss to explode on the first training iteration. Setting bias to $-4.6$ forces initial objectness to $\sigma(-4.6) \approx 0.01$, stabilizing early convergence.
5. **Decoding Loop**:
   Always vectorize the grid generation with `torch.meshgrid(..., indexing="ij")` to avoid Python `for` loops during inference.
