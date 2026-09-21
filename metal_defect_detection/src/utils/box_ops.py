"""
Bounding Box Operations: IoU, CIoU, and Non-Maximum Suppression (NMS).
"""

from typing import List, Tuple, Optional
import torch
import torchvision


def box_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise Intersection over Union (IoU) of boxes [xmin, ymin, xmax, ymax].
    
    Args:
        box1: Tensor of shape (N, 4)
        box2: Tensor of shape (M, 4)
        
    Returns:
        Tensor of shape (N, M) with pairwise IoU values.
    """
    area1 = (box1[:, 2] - box1[:, 0]).clamp(min=0) * (box1[:, 3] - box1[:, 1]).clamp(min=0)
    area2 = (box2[:, 2] - box2[:, 0]).clamp(min=0) * (box2[:, 3] - box2[:, 1]).clamp(min=0)

    lt = torch.max(box1[:, None, :2], box2[:, :2])  # [N, M, 2]
    rb = torch.min(box1[:, None, 2:], box2[..., 2:])  # [N, M, 2]

    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

    union = area1[:, None] + area2 - inter
    return inter / (union + 1e-7)


def bbox_ciou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """
    Complete Intersection over Union (CIoU) for bounding box regression.
    Accounts for overlap area, central point distance, and aspect ratio consistency.
    
    Args:
        box1: (N, 4) in [xmin, ymin, xmax, ymax]
        box2: (N, 4) in [xmin, ymin, xmax, ymax]
        
    Returns:
        CIoU values of shape (N, 1) or (N,)
    """
    # Overlap
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    w1, h1 = (b1_x2 - b1_x1).clamp(min=0), (b1_y2 - b1_y1).clamp(min=0)
    w2, h2 = (b2_x2 - b2_x1).clamp(min=0), (b2_y2 - b2_y1).clamp(min=0)

    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(min=0) * (
        torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)
    ).clamp(min=0)
    union = w1 * h1 + w2 * h2 - inter + 1e-7
    iou = inter / union

    # Smallest enclosing box
    cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
    ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
    c2 = cw**2 + ch**2 + 1e-7

    # Center distance squared
    b1_cx, b1_cy = (b1_x1 + b1_x2) / 2, (b1_y1 + b1_y2) / 2
    b2_cx, b2_cy = (b2_x1 + b2_x2) / 2, (b2_y1 + b2_y2) / 2
    rho2 = (b2_cx - b1_cx) ** 2 + (b2_cy - b1_cy) ** 2

    # Aspect ratio consistency v and weight alpha
    v = (4 / (torch.pi**2)) * torch.pow(torch.atan(w2 / (h2 + 1e-7)) - torch.atan(w1 / (h1 + 1e-7)), 2)
    with torch.no_grad():
        alpha = v / (1 - iou + v + 1e-7)

    ciou = iou - (rho2 / c2 + v * alpha)
    return ciou


def non_max_suppression(
    prediction: torch.Tensor,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    max_det: int = 300,
) -> List[torch.Tensor]:
    """
    Perform Non-Maximum Suppression (NMS) on inference predictions.
    
    Args:
        prediction: Tensor of shape (B, num_proposals, 5 + num_classes)
                    where columns are [x, y, w, h, obj_conf, class_probs...]
        conf_thres: Minimum confidence threshold.
        iou_thres: IoU threshold for NMS.
        max_det: Maximum detections per image.
        
    Returns:
        List of length B, each element is a Tensor of shape (K, 6)
        with columns: [xmin, ymin, xmax, ymax, score, class_id]
    """
    bs = prediction.shape[0]
    nc = prediction.shape[2] - 5
    output = [torch.zeros((0, 6), device=prediction.device)] * bs

    for xi, x in enumerate(prediction):
        # x is (num_proposals, 5 + nc)
        # Compute detection score: obj_conf * class_conf
        scores = x[:, 4:5] * x[:, 5:]
        max_scores, class_ids = scores.max(dim=1, keepdim=True)

        mask = max_scores.squeeze(1) > conf_thres
        x = x[mask]
        max_scores = max_scores[mask]
        class_ids = class_ids[mask]

        if not x.shape[0]:
            continue

        # Convert xywh to xyxy
        xc, yc, w, h = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
        boxes = torch.stack([xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2], dim=1)

        # Use torchvision nms
        # Add class offset to perform class-specific NMS
        c = class_ids * 4096.0
        boxes_with_offset = boxes + c
        keep = torchvision.ops.nms(boxes_with_offset, max_scores.squeeze(1), iou_thres)
        keep = keep[:max_det]

        output[xi] = torch.cat(
            [boxes[keep], max_scores[keep], class_ids[keep].float()], dim=1
        )

    return output
