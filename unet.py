"""
models/unet.py
--------------
U-Net implementation for multi-class brain tumour segmentation.

Architecture
------------
Encoder  : 4 down-sampling blocks (Conv-BN-ReLU × 2 + MaxPool)
Bottleneck: Conv-BN-ReLU × 2 with dropout
Decoder  : 4 up-sampling blocks (TransposeConv + skip concat + Conv-BN-ReLU × 2)
Head     : 1×1 Conv → num_classes logits

Channels : [64, 128, 256, 512] → bottleneck 1024

Input    : [B, 3, H, W]   (H, W must be divisible by 16)
Output   : [B, num_classes, H, W]  (raw logits)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """Two consecutive Conv-BN-ReLU layers."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DownBlock(nn.Module):
    """ConvBlock followed by 2×2 MaxPool."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch, dropout)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        skip = self.conv(x)
        return self.pool(skip), skip


class UpBlock(nn.Module):
    """
    Transposed-Conv up-sample, concatenate skip connection, then ConvBlock.
    Uses bilinear up-sample + 1×1 conv as alternative if bilinear=True.
    """

    def __init__(self, in_ch: int, out_ch: int,
                 bilinear: bool = False, dropout: float = 0.0):
        super().__init__()
        if bilinear:
            self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.proj = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.up   = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
            self.proj = nn.Identity()
        self.conv = ConvBlock(in_ch, out_ch, dropout)   # in_ch because skip doubles channels

    def forward(self, x, skip):
        x = self.proj(self.up(x))

        # Pad in case spatial dims differ by 1 pixel (odd input sizes)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)

        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


# ─────────────────────────────────────────────────────────────────────────────
# U-Net
# ─────────────────────────────────────────────────────────────────────────────

class UNet(nn.Module):
    """
    Parameters
    ----------
    in_channels  : int   — input image channels (3 for RGB / multi-modal)
    num_classes  : int   — number of segmentation classes
    base_filters : int   — filters in first encoder block (doubles each block)
    dropout      : float — dropout rate in bottleneck
    bilinear     : bool  — use bilinear up-sampling instead of transposed conv
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 4,
                 base_filters: int = 64, dropout: float = 0.3,
                 bilinear: bool = False):
        super().__init__()
        f = base_filters                 # shorthand

        # Encoder
        self.down1 = DownBlock(in_channels, f)
        self.down2 = DownBlock(f,     f * 2)
        self.down3 = DownBlock(f * 2, f * 4)
        self.down4 = DownBlock(f * 4, f * 8)

        # Bottleneck
        self.bottleneck = ConvBlock(f * 8, f * 16, dropout=dropout)

        # Decoder
        self.up4 = UpBlock(f * 16, f * 8,  bilinear=bilinear)
        self.up3 = UpBlock(f * 8,  f * 4,  bilinear=bilinear)
        self.up2 = UpBlock(f * 4,  f * 2,  bilinear=bilinear)
        self.up1 = UpBlock(f * 2,  f,      bilinear=bilinear)

        # Output head
        self.head = nn.Conv2d(f, num_classes, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x,  s1 = self.down1(x)
        x,  s2 = self.down2(x)
        x,  s3 = self.down3(x)
        x,  s4 = self.down4(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        x = self.up4(x, s4)
        x = self.up3(x, s3)
        x = self.up2(x, s2)
        x = self.up1(x, s1)

        return self.head(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = UNet(in_channels=3, num_classes=4, base_filters=64)
    x     = torch.randn(2, 3, 256, 256)
    out   = model(x)
    print(f"Input  : {x.shape}")
    print(f"Output : {out.shape}")
    print(f"Params : {model.count_parameters():,}")
    assert out.shape == (2, 4, 256, 256), "Shape mismatch!"
    print("Model sanity check passed.")
