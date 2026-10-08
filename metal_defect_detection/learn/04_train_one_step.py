#!/usr/bin/env python3
"""
Step 4: Understand the Training Loop (One Mini-Batch Step).
===========================================================
This script teaches you:
1. How a single training step works mathematically in PyTorch.
2. The 3 loss components: Box CIoU loss, Objectness BCE, and Classification BCE.
3. How backpropagation (loss.backward()) computes gradients.
4. How the AdamW optimizer updates the weights to reduce error.
"""

import sys
from pathlib import Path

# Add project root to sys.path so 'import src...' works regardless of current working directory
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from torch.utils.data import DataLoader

from src.dataset.loader import NEUDataset, yolo_collate_fn
from src.models.detector import DefectDetector
from src.training.loss import ComputeLoss


def main():
    print("=" * 70)
    print("STEP 4: HOW NEURAL NETWORKS LEARN (THE TRAINING STEP)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[1] Running training step on: {device}")

    # 1. Instantiate fresh baseline model
    model = DefectDetector.build_model(variant="M1", num_classes=6).to(device)
    model.train()

    # 2. Setup AdamW Optimizer
    learning_rate = 0.001
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0005)

    # 3. Setup Multi-Task Loss Function
    compute_loss = ComputeLoss(
        box_gain=0.05,
        cls_gain=0.5,
        obj_gain=1.0,
    )

    # 4. Fetch one real mini-batch (batch size = 4)
    data_dir = SCRIPT_DIR / "data" / "NEU-DET" / "train"
    if not data_dir.exists():
        data_dir = REPO_ROOT / "learn" / "data" / "NEU-DET" / "train"
    dataset = NEUDataset(
        image_dir=data_dir / "images",
        label_dir=data_dir / "labels",
        target_size=(200, 200),
        augment=False,
    )
    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=yolo_collate_fn)
    images, targets, _ = next(iter(loader))

    images = images.to(device)
    targets = targets.to(device)

    print("\n[2] Mini-Batch Loaded:")
    print(f"    • Images:  {images.shape} (4 images)")
    print(f"    • Targets: {targets.shape} ({len(targets)} ground-truth defect boxes)")

    # 5. Forward Pass
    # In training mode, model returns (decoded, raw_outputs)
    _decoded, raw_outputs = model(images)

    # 6. Compute Multi-Task Loss
    total_loss, loss_dict = compute_loss(raw_outputs, targets, model)

    print("\n[3] Loss Formulation (Before Weight Update):")
    print(f"    • Box CIoU Loss:       {loss_dict.get('box_loss', 0.0):.4f}  (Localization error)")
    print(f"    • Objectness BCE Loss: {loss_dict.get('obj_loss', 0.0):.4f}  (Confidence: defect vs background)")
    print(f"    • Classification BCE:  {loss_dict.get('cls_loss', 0.0):.4f}  (Defect category error)")
    print("    --------------------------------------------------")
    print(f"    • TOTAL COMBINED LOSS: {total_loss.item():.4f}")

    # 7. Backward Pass: Backpropagation computes dLoss / dWeights
    optimizer.zero_grad()
    total_loss.backward()

    # Calculate average gradient magnitude
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
    print("\n[4] Backpropagation:")
    print("    • Gradients calculated via Chain Rule (dLoss/dW).")
    print(f"    • Gradient Norm magnitude: {grad_norm.item():.4f}")

    # 8. Optimizer Step: W_new = W_old - lr * gradients
    optimizer.step()
    print("\n[5] Optimizer Step:")
    print(f"    • Updated {sum(p.numel() for p in model.parameters()):,} weights using AdamW.")

    # 9. Verify loss decreases on the same batch after update
    with torch.no_grad():
        _, raw_outputs_after = model(images)
        new_loss, _new_loss_dict = compute_loss(raw_outputs_after, targets, model)

    print("\n[6] Immediate Effect of Weight Update on Same Batch:")
    print(f"    • Loss before step: {total_loss.item():.4f}")
    print(f"    • Loss after step:  {new_loss.item():.4f}")
    diff = total_loss.item() - new_loss.item()
    if diff > 0:
        print(f"    -> Loss decreased by {diff:.4f} (The model learned!)")
    else:
        print(f"    -> Loss difference: {diff:+.4f} (Expected stochastic step fluctuation)")

    print("\n" + "=" * 70)
    print("STEP 4 COMPLETE: You now understand the full PyTorch training loop!")
    print("In full training (scripts/train.py), this exact step repeats thousands of times across 50 epochs.")
    print("=" * 70)

if __name__ == "__main__":
    main()
