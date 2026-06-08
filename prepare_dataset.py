"""
prepare_dataset.py
------------------
Two modes:
  1. REAL   : Downloads BraTS 2023 from Kaggle and converts NIfTI -> PNG slices.
  2. SYNTHETIC: Generates a synthetic brain MRI dataset that mirrors the
               BraTS structure (T1, T2, FLAIR channels + 3-class mask).
               Use this to test the pipeline without downloading 70 GB.

Usage
-----
  # Synthetic (quick test)
  python data/prepare_dataset.py --mode synthetic --n_samples 200 --out_dir data/processed

  # Real BraTS (requires kaggle API key configured)
  python data/prepare_dataset.py --mode real --brats_dir /path/to/brats2023 --out_dir data/processed
"""

import os
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
import json
import random


# ─────────────────────────────────────────────
# Synthetic generation helpers
# ─────────────────────────────────────────────

def _ellipse_mask(shape, cx, cy, rx, ry, angle=0.0):
    """Return boolean mask of a rotated ellipse."""
    H, W = shape
    Y, X = np.ogrid[:H, :W]
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    Xr = cos_a * (X - cx) + sin_a * (Y - cy)
    Yr = -sin_a * (X - cx) + cos_a * (Y - cy)
    return (Xr / rx) ** 2 + (Yr / ry) ** 2 <= 1.0


def generate_synthetic_sample(H=240, W=240, rng=None):
    """
    Generate one synthetic (image, mask) pair.

    Image  : float32 [H, W, 3]  — channels simulate T1, T2, FLAIR
    Mask   : uint8   [H, W]     — 0=background, 1=edema, 2=tumor-core, 3=enhancing-tumor
    """
    if rng is None:
        rng = np.random.default_rng()

    img = np.zeros((H, W, 3), dtype=np.float32)

    # ── skull / brain region ──────────────────────────────────────────────
    cx, cy = W // 2 + rng.integers(-10, 10), H // 2 + rng.integers(-10, 10)
    brain_rx, brain_ry = rng.integers(80, 100), rng.integers(90, 110)
    brain = _ellipse_mask((H, W), cx, cy, brain_rx, brain_ry, angle=rng.uniform(0, 0.3))

    # T1, T2, FLAIR slightly different intensities for brain tissue
    for ch, base in enumerate([0.55, 0.50, 0.45]):
        noise = rng.normal(0, 0.04, (H, W)).astype(np.float32)
        img[:, :, ch] += brain.astype(np.float32) * (base + noise)

    # ── tumor sub-regions ─────────────────────────────────────────────────
    mask = np.zeros((H, W), dtype=np.uint8)

    # Place tumor within brain
    t_cx = cx + rng.integers(-30, 30)
    t_cy = cy + rng.integers(-30, 30)

    # Edema (label 1)  — largest ring
    edema_rx = rng.integers(20, 35)
    edema_ry = rng.integers(18, 30)
    edema_angle = rng.uniform(0, np.pi)
    edema = _ellipse_mask((H, W), t_cx, t_cy, edema_rx, edema_ry, edema_angle)
    edema &= brain
    mask[edema] = 1

    # Tumor core (label 2) — smaller inside edema
    core_rx = rng.integers(10, edema_rx - 4)
    core_ry = rng.integers(8, edema_ry - 4)
    core = _ellipse_mask((H, W), t_cx, t_cy, core_rx, core_ry, edema_angle)
    core &= brain
    mask[core] = 2

    # Enhancing tumor (label 3) — smallest bright center
    enh_rx = max(3, core_rx - rng.integers(3, 6))
    enh_ry = max(3, core_ry - rng.integers(3, 6))
    enh = _ellipse_mask((H, W), t_cx, t_cy, enh_rx, enh_ry, edema_angle)
    enh &= brain
    mask[enh] = 3

    # Boost intensities inside tumor regions per channel
    for region, intensities in [
        (edema,  [0.65, 0.80, 0.85]),   # edema bright on T2/FLAIR
        (core,   [0.70, 0.60, 0.55]),   # core bright on T1
        (enh,    [0.90, 0.70, 0.65]),   # enhancing very bright T1
    ]:
        for ch, val in enumerate(intensities):
            img[:, :, ch] += region.astype(np.float32) * val * 0.4

    # Clip and add mild global noise
    img = np.clip(img + rng.normal(0, 0.01, img.shape).astype(np.float32), 0.0, 1.0)

    return img, mask


def save_synthetic_dataset(out_dir: Path, n_samples: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    splits = {"train": int(0.70 * n_samples),
              "val":   int(0.15 * n_samples),
              "test":  n_samples - int(0.70 * n_samples) - int(0.15 * n_samples)}

    meta = []
    idx = 0
    for split, count in splits.items():
        split_dir = out_dir / split
        (split_dir / "images").mkdir(parents=True, exist_ok=True)
        (split_dir / "masks").mkdir(parents=True, exist_ok=True)

        for _ in range(count):
            img, mask = generate_synthetic_sample(rng=rng)

            # Save image as 3-channel PNG (uint8 0-255)
            img_u8 = (img * 255).astype(np.uint8)
            img_path = split_dir / "images" / f"sample_{idx:04d}.png"
            Image.fromarray(img_u8).save(img_path)

            # Save mask as single-channel PNG
            mask_path = split_dir / "masks" / f"sample_{idx:04d}.png"
            Image.fromarray(mask).save(mask_path)

            meta.append({"id": idx, "split": split,
                         "image": str(img_path), "mask": str(mask_path)})
            idx += 1

    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[SYNTHETIC] Saved {n_samples} samples → {out_dir}")
    print(f"  train={splits['train']}  val={splits['val']}  test={splits['test']}")
    print(f"  metadata → {out_dir / 'metadata.json'}")


# ─────────────────────────────────────────────
# Real BraTS converter
# ─────────────────────────────────────────────

def convert_brats_to_png(brats_dir: Path, out_dir: Path, slices_per_volume=10):
    """
    Converts BraTS NIfTI volumes to 2-D PNG slices.
    Requires: nibabel  (pip install nibabel)

    brats_dir layout expected:
      brats_dir/
        BraTS2023_*/
          *_t1.nii.gz
          *_t2.nii.gz
          *_flair.nii.gz
          *_seg.nii.gz
    """
    try:
        import nibabel as nib
    except ImportError:
        raise ImportError("nibabel is required for real BraTS processing: pip install nibabel")

    cases = sorted([d for d in brats_dir.iterdir() if d.is_dir()])
    random.shuffle(cases)
    n = len(cases)
    train_end = int(0.70 * n)
    val_end   = int(0.85 * n)

    def get_split(i):
        if i < train_end:   return "train"
        if i < val_end:     return "val"
        return "test"

    meta = []
    idx = 0
    for i, case in enumerate(cases):
        split = get_split(i)
        (out_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (out_dir / split / "masks").mkdir(parents=True, exist_ok=True)

        t1    = nib.load(next(case.glob("*_t1.nii.gz"))).get_fdata()
        t2    = nib.load(next(case.glob("*_t2.nii.gz"))).get_fdata()
        flair = nib.load(next(case.glob("*_flair.nii.gz"))).get_fdata()
        seg   = nib.load(next(case.glob("*_seg.nii.gz"))).get_fdata().astype(np.uint8)

        D = t1.shape[2]
        # Pick slices that contain tumor
        tumor_slices = [s for s in range(D) if seg[:, :, s].max() > 0]
        chosen = random.sample(tumor_slices, min(slices_per_volume, len(tumor_slices)))

        for sl in chosen:
            # Normalise each modality to [0,1]
            def norm(v): return (v - v.min()) / (v.max() - v.min() + 1e-8)
            stack = np.stack([norm(t1[:, :, sl]),
                              norm(t2[:, :, sl]),
                              norm(flair[:, :, sl])], axis=-1)
            img_u8 = (stack * 255).astype(np.uint8)

            img_path  = out_dir / split / "images" / f"sample_{idx:05d}.png"
            mask_path = out_dir / split / "masks"  / f"sample_{idx:05d}.png"
            Image.fromarray(img_u8).save(img_path)
            Image.fromarray(seg[:, :, sl]).save(mask_path)
            meta.append({"id": idx, "case": case.name, "slice": sl,
                         "split": split, "image": str(img_path), "mask": str(mask_path)})
            idx += 1

    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[REAL BraTS] Saved {idx} slices from {n} volumes → {out_dir}")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",      choices=["synthetic", "real"], default="synthetic")
    parser.add_argument("--out_dir",   default="data/processed")
    parser.add_argument("--n_samples", type=int, default=300,
                        help="Number of synthetic samples (ignored for real mode)")
    parser.add_argument("--brats_dir", default=None,
                        help="Path to BraTS 2023 root directory (real mode only)")
    parser.add_argument("--slices",    type=int, default=10,
                        help="Slices per volume to extract (real mode only)")
    parser.add_argument("--seed",      type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "synthetic":
        save_synthetic_dataset(out_dir, args.n_samples, seed=args.seed)
    else:
        if args.brats_dir is None:
            raise ValueError("--brats_dir required for real mode")
        convert_brats_to_png(Path(args.brats_dir), out_dir, args.slices)
