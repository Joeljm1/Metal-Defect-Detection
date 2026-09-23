"""
Multi-Task Loss Function for Metal Defect Detection.

Combines:
1. Bounding box regression loss (CIoU / GIoU)
2. Objectness confidence loss (Binary Cross-Entropy with Logits)
3. Defect classification loss (Binary Cross-Entropy with Logits)
"""

from typing import List, Tuple, Dict, Any
import torch
import torch.nn as nn

from src.utils.box_ops import bbox_ciou


class ComputeLoss(nn.Module):
    """
    Computes multi-scale YOLO loss with CIoU box regression.
    """

    def __init__(
        self,
        hyp: Dict[str, float] | None = None,
        box_gain: float = 0.05,
        cls_gain: float = 0.5,
        obj_gain: float = 1.0,
        obj_pos_weight: float = 25.0,
        anchor_threshold: float = 4.0,
    ):
        super().__init__()
        self.box_gain = box_gain
        self.cls_gain = cls_gain
        self.obj_gain = obj_gain
        self.obj_pos_weight = obj_pos_weight
        self.anchor_threshold = anchor_threshold

        self.bce_cls = nn.BCEWithLogitsLoss(reduction="mean")
        self.bce_obj = None  # Instantiated with device-specific pos_weight in forward()

        # Balance weights for P3, P4, P5
        self.balance = [4.0, 1.0, 0.4]

    def forward(
        self,
        predictions: List[torch.Tensor] | Tuple[torch.Tensor, List[torch.Tensor]],
        targets: torch.Tensor,
        model: nn.Module,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            predictions: List of 3 raw prediction tensors [(B, na, ny, nx, no), ...]
                         or tuple (decoded, raw_predictions) when model is in eval mode.
            targets: Tensor of shape (total_boxes, 6): [image_idx, class_id, xc, yc, w, h] (normalized)
            model: DefectDetector instance
            
        Returns:
            total_loss: scalar loss tensor
            loss_items: dictionary of itemized losses for logging
        """
        if isinstance(predictions, tuple):
            predictions = predictions[1]

        # Defensive filtering: remove targets with out-of-bounds class IDs
        num_classes = getattr(getattr(model, "head", None), "num_classes", 6)
        if targets.numel() > 0:
            valid_cls_mask = (targets[:, 1] >= 0) & (targets[:, 1] < num_classes)
            targets = targets[valid_cls_mask]

        device = targets.device if targets.numel() > 0 else predictions[0].device
        if self.bce_obj is None or self.bce_obj.pos_weight.device != device:
            pos_weight = torch.tensor([self.obj_pos_weight], device=device)
            self.bce_obj = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="mean")

        lbox = torch.zeros(1, device=device)
        lobj = torch.zeros(1, device=device)
        lcls = torch.zeros(1, device=device)

        # Build targets for each scale
        tcls, tbox, indices, anchors = self._build_targets(predictions, targets, model)

        for i, pred in enumerate(predictions):
            b, a, gj, gi = indices[i]
            tobj = torch.zeros_like(pred[..., 4], device=device)

            nb = b.shape[0]  # number of matching targets
            if nb:
                # Prediction subset matching targets
                ps = pred[b, a, gj, gi]

                # Regression: decode predicted box
                pxy = ps[:, 0:2].sigmoid() * 2.0 - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2.0) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), dim=1)

                # CIoU loss: 1.0 - ciou
                # Convert both pbox and tbox to xyxy
                p_x1y1 = pbox[:, :2] - pbox[:, 2:] / 2
                p_x2y2 = pbox[:, :2] + pbox[:, 2:] / 2
                p_xyxy = torch.cat([p_x1y1, p_x2y2], dim=1)

                t_x1y1 = tbox[i][:, :2] - tbox[i][:, 2:] / 2
                t_x2y2 = tbox[i][:, :2] + tbox[i][:, 2:] / 2
                t_xyxy = torch.cat([t_x1y1, t_x2y2], dim=1)

                ciou = bbox_ciou(p_xyxy, t_xyxy)
                lbox += (1.0 - ciou).mean()

                # Target objectness is CIoU score clamped
                tobj[b, a, gj, gi] = ciou.detach().clamp(0).type(tobj.dtype)

                # Classification loss
                if ps.shape[1] > 5:
                    tc = torch.zeros_like(ps[:, 5:], device=device)
                    tc[torch.arange(nb, device=device), tcls[i]] = 1.0
                    lcls += self.bce_cls(ps[:, 5:], tc)

            # Objectness loss across full grid
            lobj += self.bce_obj(pred[..., 4], tobj) * self.balance[i]

        total_loss = (lbox * self.box_gain) + (lobj * self.obj_gain) + (lcls * self.cls_gain)

        loss_dict = {
            "loss_total": float(total_loss.item()),
            "loss_box": float(lbox.item()),
            "loss_obj": float(lobj.item()),
            "loss_cls": float(lcls.item()),
        }
        return total_loss, loss_dict

    def _build_targets(
        self, predictions: List[torch.Tensor], targets: torch.Tensor, model: nn.Module
    ):
        """
        Assign ground-truth targets to detection grid cells and anchors.
        """
        device = targets.device if targets.numel() > 0 else predictions[0].device
        nl = len(predictions)
        na = model.head.na
        nt = targets.shape[0]

        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=device)
        ai = torch.arange(na, device=device).float().view(na, 1).repeat(1, nt)
        targets_rep = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)

        g = 0.5  # bias
        off = (
            torch.tensor(
                [
                    [0, 0],
                    [1, 0],
                    [0, 1],
                    [-1, 0],
                    [0, -1],
                ],
                device=device,
            ).float()
            * g
        )

        for i in range(nl):
            anchors_layer = model.head.anchors[i].to(device) / model.head.strides[i]
            shape = predictions[i].shape  # (B, na, ny, nx, no)
            gain[2:6] = torch.tensor([shape[3], shape[2], shape[3], shape[2]], device=device)

            t = targets_rep * gain
            if nt:
                r = t[:, :, 4:6] / anchors_layer[:, None]
                j = torch.max(r, 1 / r).max(2)[0] < self.anchor_threshold
                t = t[j]

                # Offsets
                gxy = t[:, 2:4]
                gxi = gain[[2, 3]] - gxy
                j, k = ((gxy % 1.0 < g) & (gxy > 1.0)).T
                l, m = ((gxi % 1.0 < g) & (gxi > 1.0)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets_rep[0]
                offsets = 0

            b, c = t[:, :2].long().T
            gxy = t[:, 2:4]
            gwh = t[:, 4:6]
            gij = (gxy - offsets).long()
            gi, gj = gij.T

            a = t[:, 6].long()
            indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))
            tbox.append(torch.cat((gxy - gij, gwh), 1))
            anch.append(anchors_layer[a])
            tcls.append(c)

        return tcls, tbox, indices, anch
