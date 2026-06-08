"""
train.py
--------
Full training pipeline for U-Net brain tumour segmentation.

Features
--------
• Combined Dice + Focal loss (handles <2% tumour-voxel imbalance)
• OneCycleLR scheduler with warm-up
• Best-checkpoint saving on val mean Dice
• TensorBoard / W&B logging
• Test-Time Augmentation (TTA) at inference

Usage
-----
  # Train on synthetic data (quick test)
  python train.py --data_dir data/processed --epochs 50 --batch_size 8

  # Full run on BraTS
  python train.py --data_dir data/processed --epochs 150 --batch_size 16 \
                  --img_size 256 --base_filters 64 --use_wandb
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.tensorboard import SummaryWriter

from models.unet import UNet
from utils.dataset import BraTSDataset, get_train_transforms, get_val_transforms
from utils.losses import DiceFocalLoss
from utils.metrics import SegmentationMetrics


# ─────────────────────────────────────────────────────────────────────────────
# TTA helpers
# ─────────────────────────────────────────────────────────────────────────────

def tta_predict(model: nn.Module, image: torch.Tensor,
                num_classes: int) -> torch.Tensor:
    """
    8-fold Test-Time Augmentation:
      original + 3 rotations × (normal + horizontal flip)

    image : [B, C, H, W]
    Returns averaged softmax probabilities [B, num_classes, H, W]
    """
    import torch.nn.functional as F

    preds = []
    for k in range(4):                            # 0, 90, 180, 270 degrees
        for flip in [False, True]:
            aug = torch.rot90(image, k, dims=[2, 3])
            if flip:
                aug = torch.flip(aug, dims=[3])

            with torch.no_grad():
                logits = model(aug)               # [B, C, H, W]
            probs = F.softmax(logits, dim=1)

            # Reverse augmentation on probabilities
            if flip:
                probs = torch.flip(probs, dims=[3])
            probs = torch.rot90(probs, -k, dims=[2, 3])
            preds.append(probs)

    return torch.stack(preds).mean(dim=0)         # [B, C, H, W]


# ─────────────────────────────────────────────────────────────────────────────
# Training / validation loops
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, scheduler,
                    criterion, device, scaler):
    model.train()
    total_loss = dice_l = focal_l = 0.0

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device,  non_blocking=True)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            logits = model(images)
            loss, d_loss, f_loss = criterion(logits, masks)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()

        total_loss += loss.item()
        dice_l     += d_loss.item()
        focal_l    += f_loss.item()

    n = len(loader)
    return total_loss / n, dice_l / n, focal_l / n


@torch.no_grad()
def validate(model, loader, criterion, device, num_classes, use_tta=False):
    model.eval()
    metrics  = SegmentationMetrics(num_classes, ignore_background=True)
    val_loss = 0.0

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device,  non_blocking=True)

        if use_tta:
            probs  = tta_predict(model, images, num_classes)
            preds  = probs.argmax(dim=1)
            logits = torch.log(probs + 1e-8)          # pseudo-logits for loss
        else:
            logits = model(images)
            preds  = logits.argmax(dim=1)

        loss, _, _ = criterion(logits, masks)
        val_loss  += loss.item()
        metrics.update(preds, masks)

    results = metrics.compute()
    results["loss"] = val_loss / len(loader)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    # ── reproducibility ──────────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # ── data ─────────────────────────────────────────────────────────────────
    train_ds = BraTSDataset(args.data_dir, "train",
                            transform=get_train_transforms(args.img_size),
                            num_classes=args.num_classes)
    val_ds   = BraTSDataset(args.data_dir, "val",
                            transform=get_val_transforms(args.img_size),
                            num_classes=args.num_classes)
    test_ds  = BraTSDataset(args.data_dir, "test",
                            transform=get_val_transforms(args.img_size),
                            num_classes=args.num_classes)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=args.workers,
                              pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.workers,
                              pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size,
                              shuffle=False, num_workers=args.workers,
                              pin_memory=True)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    # ── model ─────────────────────────────────────────────────────────────────
    model = UNet(in_channels=3, num_classes=args.num_classes,
                 base_filters=args.base_filters, dropout=args.dropout).to(device)
    print(f"Parameters: {model.count_parameters():,}")

    # ── loss ─────────────────────────────────────────────────────────────────
    criterion = DiceFocalLoss(num_classes=args.num_classes,
                              gamma=args.focal_gamma,
                              lambda_dice=0.5, lambda_focal=0.5,
                              ignore_bg=True).to(device)

    # ── optimiser + scheduler ────────────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=args.lr,
                      weight_decay=args.weight_decay)
    scheduler = OneCycleLR(optimizer, max_lr=args.lr,
                           steps_per_epoch=len(train_loader),
                           epochs=args.epochs,
                           pct_start=0.1)

    # ── AMP scaler ────────────────────────────────────────────────────────────
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    # ── logging ───────────────────────────────────────────────────────────────
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(out_dir / "tb_logs"))

    if args.use_wandb:
        import wandb
        wandb.init(project="unet-brats", config=vars(args))

    # ── training loop ────────────────────────────────────────────────────────
    best_dice   = 0.0
    best_ckpt   = out_dir / "best_model.pth"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_dice_l, tr_focal_l = train_one_epoch(
            model, train_loader, optimizer, scheduler,
            criterion, device, scaler)

        val_metrics = validate(model, val_loader, criterion,
                               device, args.num_classes, use_tta=False)

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"Loss {tr_loss:.4f} (dice {tr_dice_l:.4f} focal {tr_focal_l:.4f}) | "
            f"Val loss {val_metrics['loss']:.4f} | "
            f"Val Dice {val_metrics['mean_dice']:.4f} | "
            f"Val IoU {val_metrics['mean_iou']:.4f} | "
            f"{elapsed:.1f}s"
        )

        # TensorBoard
        writer.add_scalar("Loss/train",     tr_loss,                    epoch)
        writer.add_scalar("Loss/val",       val_metrics["loss"],        epoch)
        writer.add_scalar("Dice/val",       val_metrics["mean_dice"],   epoch)
        writer.add_scalar("IoU/val",        val_metrics["mean_iou"],    epoch)
        writer.add_scalar("LR",
            optimizer.param_groups[0]["lr"], epoch)

        if args.use_wandb:
            import wandb
            wandb.log({"epoch": epoch, "train_loss": tr_loss,
                       "val_dice": val_metrics["mean_dice"],
                       "val_iou":  val_metrics["mean_iou"]})

        # Checkpoint
        if val_metrics["mean_dice"] > best_dice:
            best_dice = val_metrics["mean_dice"]
            torch.save({"epoch": epoch,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "best_dice": best_dice,
                        "args": vars(args)},
                       best_ckpt)
            print(f"  ✓ New best Dice: {best_dice:.4f} → saved to {best_ckpt}")

    # ── test evaluation with TTA ──────────────────────────────────────────────
    print("\n--- Test Evaluation (with 8-fold TTA) ---")
    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    test_metrics = validate(model, test_loader, criterion,
                            device, args.num_classes, use_tta=True)
    print(f"Test mean Dice : {test_metrics['mean_dice']:.4f}")
    print(f"Test mean IoU  : {test_metrics['mean_iou']:.4f}")
    print(f"Per-class Dice : {[f'{d:.3f}' for d in test_metrics['dice_per_class']]}")
    print(f"Per-class IoU  : {[f'{i:.3f}' for i in test_metrics['iou_per_class']]}")

    writer.close()
    if args.use_wandb:
        import wandb
        wandb.finish()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="U-Net Brain Tumour Segmentation")

    # Data
    parser.add_argument("--data_dir",    default="data/processed")
    parser.add_argument("--out_dir",     default="outputs")
    parser.add_argument("--img_size",    type=int, default=256)
    parser.add_argument("--num_classes", type=int, default=4)

    # Model
    parser.add_argument("--base_filters", type=int,   default=64)
    parser.add_argument("--dropout",      type=float, default=0.3)

    # Training
    parser.add_argument("--epochs",       type=int,   default=100)
    parser.add_argument("--batch_size",   type=int,   default=8)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--focal_gamma",  type=float, default=2.0)
    parser.add_argument("--workers",      type=int,   default=4)
    parser.add_argument("--seed",         type=int,   default=42)

    # Logging
    parser.add_argument("--use_wandb", action="store_true")

    args = parser.parse_args()
    main(args)
