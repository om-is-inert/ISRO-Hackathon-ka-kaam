"""
Model 2: Refinement Network (U-Net + CBAM Attention)

Takes the coarse interpolation from Model 1 along with all intermediate
representations (warped frames, flow maps, blend mask, original frames)
and learns a RESIDUAL correction to produce a sharp, artifact-free output.

Key design decisions:
  - Residual learning: output = coarse + small_correction
  - CBAM attention: focuses on cloud edges and thermal boundaries
  - GroupNorm: more stable than BatchNorm for small batch sizes (Colab)
  - Tanh output head: bounded residuals for training stability

Optimized for free-tier GPU (T4 15GB):
  - base_channels reduced 48→32 (~55% fewer params)
  - Gradient checkpointing on bottleneck to save VRAM
  - Encoder uses 1 ResBlock instead of 2 (faster, similar quality)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class ChannelAttention(nn.Module):
    """Channel attention — learns which feature channels are most important."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
        )
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return torch.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """Spatial attention — learns where to focus (e.g., cloud boundaries)."""
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        return torch.sigmoid(self.conv(combined))


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.
    Sequentially applies channel attention then spatial attention.
    This helps the refinement model focus on:
      - Important feature channels (channel attention)
      - Important spatial regions like cloud edges (spatial attention)
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention()
    
    def forward(self, x):
        x = x * self.channel_att(x)
        x = x * self.spatial_att(x)
        return x


class ResBlock(nn.Module):
    """Residual block with CBAM attention and GroupNorm."""
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(min(8, channels), channels),
            nn.PReLU(channels),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(min(8, channels), channels),
        )
        self.attn = CBAM(channels, reduction=max(1, channels // 4))
        self.act = nn.PReLU(channels)
    
    def forward(self, x):
        residual = self.attn(self.block(x))
        return self.act(x + residual)


class RefinementNet(nn.Module):
    """
    Model 2: U-Net with CBAM attention for refining coarse interpolation.
    
    Learns a residual correction: final = coarse + learned_residual
    
    Input channels (8 total for grayscale TIR):
      - Coarse interpolated frame (1 ch)
      - Warped frame t₀ (1 ch)
      - Warped frame t₁ (1 ch)
      - Original frame t₀ (1 ch)
      - Original frame t₁ (1 ch)
      - Optical flow t₀→t (2 ch)
      - Blend mask (1 ch)
    
    Output:
      - Residual correction (1 ch), scaled for stability
      - Final refined frame = coarse + residual
    
    Optimized: base_channels=32 (was 48), 1 ResBlock per stage (was 2),
    gradient checkpointing on bottleneck.
    """
    def __init__(self, in_channels=8, base_channels=32):
        super().__init__()
        C = base_channels

        # ---- Encoder (lighter: 1 ResBlock per stage) ----
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, C, 3, padding=1),
            nn.PReLU(C),
            ResBlock(C),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(C, C * 2, 3, stride=2, padding=1),
            nn.PReLU(C * 2),
            ResBlock(C * 2),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(C * 2, C * 4, 3, stride=2, padding=1),
            nn.PReLU(C * 4),
            ResBlock(C * 4),
        )

        # ---- Bottleneck (gradient-checkpointed to save VRAM) ----
        self.bottleneck = nn.Sequential(
            nn.Conv2d(C * 4, C * 4, 3, stride=2, padding=1),
            nn.PReLU(C * 4),
            ResBlock(C * 4),
            ResBlock(C * 4),
        )

        # ---- Decoder with skip connections ----
        self.up3 = nn.ConvTranspose2d(C * 4, C * 4, 2, stride=2)
        self.dec3 = nn.Sequential(
            ResBlock(C * 8),  # concat with enc3
            nn.Conv2d(C * 8, C * 4, 1),
            nn.PReLU(C * 4),
        )

        self.up2 = nn.ConvTranspose2d(C * 4, C * 2, 2, stride=2)
        self.dec2 = nn.Sequential(
            ResBlock(C * 4),  # concat with enc2
            nn.Conv2d(C * 4, C * 2, 1),
            nn.PReLU(C * 2),
        )

        self.up1 = nn.ConvTranspose2d(C * 2, C, 2, stride=2)
        self.dec1 = nn.Sequential(
            ResBlock(C * 2),  # concat with enc1
            nn.Conv2d(C * 2, C, 1),
            nn.PReLU(C),
        )

        # ---- Output head — learns RESIDUAL correction ----
        self.head = nn.Sequential(
            nn.Conv2d(C, C // 2, 3, padding=1),
            nn.PReLU(C // 2),
            nn.Conv2d(C // 2, 1, 3, padding=1),
            nn.Tanh(),  # Output in [-1, 1]
        )

        # Learnable residual scale (starts small for stability)
        self.residual_scale = nn.Parameter(torch.tensor(0.05))

        # Whether to use gradient checkpointing (enabled during training)
        self.use_checkpointing = False

    def _bottleneck_forward(self, x):
        """Bottleneck forward — wrapped for gradient checkpointing."""
        return self.bottleneck(x)

    def forward(self, coarse_frame, warped0, warped1, img0, img1, flow, mask):
        """
        Args:
            coarse_frame: (B, 1, H, W) — Model 1's output
            warped0: (B, 1, H, W) — img0 warped to target time
            warped1: (B, 1, H, W) — img1 warped to target time
            img0: (B, 1, H, W) — original frame at t=0
            img1: (B, 1, H, W) — original frame at t=1
            flow: (B, 2, H, W) — optical flow (first 2 channels)
            mask: (B, 1, H, W) — blend mask from Model 1
        
        Returns:
            refined: (B, 1, H, W) — final refined frame
            residual: (B, 1, H, W) — learned correction
        """
        # Concatenate all inputs: 1+1+1+1+1+2+1 = 8 channels
        x = torch.cat([
            coarse_frame,
            warped0,
            warped1,
            img0,
            img1,
            flow[:, :2],  # Use flow_0→t (first 2 channels)
            mask
        ], dim=1)

        # Encoder
        e1 = self.enc1(x)     # (B, C, H, W)
        e2 = self.enc2(e1)    # (B, 2C, H/2, W/2)
        e3 = self.enc3(e2)    # (B, 4C, H/4, W/4)

        # Bottleneck (with optional gradient checkpointing to save VRAM)
        if self.use_checkpointing and self.training:
            b = checkpoint(self._bottleneck_forward, e3, use_reentrant=False)
        else:
            b = self._bottleneck_forward(e3)

        # Decoder with skip connections
        d3 = self.up3(b)
        # Handle size mismatch from odd dimensions
        if d3.shape != e3.shape:
            d3 = F.interpolate(d3, size=e3.shape[2:], mode='bilinear', align_corners=True)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        if d2.shape != e2.shape:
            d2 = F.interpolate(d2, size=e2.shape[2:], mode='bilinear', align_corners=True)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        if d1.shape != e1.shape:
            d1 = F.interpolate(d1, size=e1.shape[2:], mode='bilinear', align_corners=True)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        # Residual output
        residual = self.head(d1) * self.residual_scale

        # Final = Coarse + Learned Correction
        refined = torch.clamp(coarse_frame + residual, 0.0, 1.0)

        return refined, residual

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
