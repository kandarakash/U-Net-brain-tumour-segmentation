"""
utils/metrics.py
----------------
Dice score, IoU, and per-class metric tracking for segmentation.
"""

import torch
import numpy as np


def dice_score(pred: torch.Tensor, target: torch.Tensor,
               num_classes: int, smooth: float = 1e-6) -> torch.Tensor:
    """
    Compute per-class Dice score.

    Parameters
    ----------
    pred        : LongTensor [B, H, W]  — predicted class indices
    target      : LongTensor [B, H, W]  — ground-truth class indices
    num_classes : int
    smooth      : float  — Laplace smoothing

    Returns
    -------
    dice : FloatTensor [num_classes]  — per-class Dice
    """
    dice = torch.zeros(num_classes, device=pred.device)
    for c in range(num_classes):
        pred_c   = (pred   == c).float()
        target_c = (target == c).float()
        intersection = (pred_c * target_c).sum()
        dice[c] = (2.0 * intersection + smooth) / (pred_c.sum() + target_c.sum() + smooth)
    return dice


def iou_score(pred: torch.Tensor, target: torch.Tensor,
              num_classes: int, smooth: float = 1e-6) -> torch.Tensor:
    """Per-class Intersection-over-Union."""
    iou = torch.zeros(num_classes, device=pred.device)
    for c in range(num_classes):
        pred_c   = (pred   == c).float()
        target_c = (target == c).float()
        intersection = (pred_c * target_c).sum()
        union        = pred_c.sum() + target_c.sum() - intersection
        iou[c] = (intersection + smooth) / (union + smooth)
    return iou


class SegmentationMetrics:
    """Running accumulator for segmentation metrics across batches."""

    def __init__(self, num_classes: int, ignore_background: bool = True):
        self.num_classes       = num_classes
        self.ignore_background = ignore_background
        self.reset()

    def reset(self):
        self._dice_sum = np.zeros(self.num_classes)
        self._iou_sum  = np.zeros(self.num_classes)
        self._count    = 0

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """
        pred, target : LongTensor [B, H, W]
        """
        d = dice_score(pred, target, self.num_classes).cpu().numpy()
        i = iou_score(pred,  target, self.num_classes).cpu().numpy()
        self._dice_sum += d
        self._iou_sum  += i
        self._count    += 1

    def compute(self):
        if self._count == 0:
            return {}
        dice_per_class = self._dice_sum / self._count
        iou_per_class  = self._iou_sum  / self._count

        start = 1 if self.ignore_background else 0
        return {
            "mean_dice":     float(dice_per_class[start:].mean()),
            "mean_iou":      float(iou_per_class[start:].mean()),
            "dice_per_class": dice_per_class.tolist(),
            "iou_per_class":  iou_per_class.tolist(),
        }
