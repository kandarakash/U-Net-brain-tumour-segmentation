"""
utils/dataset.py
----------------
PyTorch Dataset for brain tumour segmentation.
Works with the processed PNG slices produced by data/prepare_dataset.py.
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ─────────────────────────────────────────────────────────────────────────────
# Augmentation pipelines
# ─────────────────────────────────────────────────────────────────────────────

def get_train_transforms(img_size: int = 256):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                           rotate_limit=15, p=0.5,
                           border_mode=0),
        A.ElasticTransform(alpha=120, sigma=120 * 0.05,
                           alpha_affine=120 * 0.03, p=0.3),
        A.GridDistortion(p=0.2),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.2,
                                   contrast_limit=0.2, p=0.4),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_val_transforms(img_size: int = 256):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class BraTSDataset(Dataset):
    """
    Loads (image, mask) pairs from the processed directory.

    Parameters
    ----------
    data_dir : str | Path
        Root of processed dataset (contains train/val/test sub-dirs).
    split    : str
        One of 'train', 'val', 'test'.
    transform : albumentations Compose | None
        Augmentation / normalisation pipeline.
    num_classes : int
        Number of segmentation classes (default 4: BG + edema + core + enhancing).
    """

    def __init__(self, data_dir, split="train", transform=None, num_classes=4):
        self.data_dir   = Path(data_dir)
        self.split      = split
        self.transform  = transform
        self.num_classes = num_classes

        # Collect samples from metadata.json
        meta_path = self.data_dir / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                all_meta = json.load(f)
            self.samples = [m for m in all_meta if m["split"] == split]
        else:
            # Fallback: scan directory directly
            img_dir  = self.data_dir / split / "images"
            mask_dir = self.data_dir / split / "masks"
            img_paths = sorted(img_dir.glob("*.png"))
            self.samples = [
                {"image": str(p), "mask": str(mask_dir / p.name)}
                for p in img_paths
            ]

        if len(self.samples) == 0:
            raise RuntimeError(f"No samples found for split='{split}' in {data_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        image = np.array(Image.open(sample["image"]).convert("RGB"), dtype=np.uint8)
        mask  = np.array(Image.open(sample["mask"]).convert("L"),   dtype=np.int64)

        # Clamp mask labels to [0, num_classes - 1]
        mask = np.clip(mask, 0, self.num_classes - 1)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]          # Tensor [3, H, W]
            mask  = torch.tensor(augmented["mask"], dtype=torch.long)
        else:
            image = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32) / 255.0
            mask  = torch.tensor(mask, dtype=torch.long)

        return image, mask

    def class_weights(self):
        """Compute inverse-frequency class weights from the full split (slow on large datasets)."""
        counts = torch.zeros(self.num_classes)
        for _, mask in self:
            for c in range(self.num_classes):
                counts[c] += (mask == c).sum()
        freq   = counts / counts.sum()
        weights = 1.0 / (freq + 1e-6)
        return weights / weights.sum() * self.num_classes
