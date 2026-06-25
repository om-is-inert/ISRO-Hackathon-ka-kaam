"""
Model 1: Flow-Based Interpolator (Modified RIFE Architecture)

Multi-scale optical flow estimation with coarse-to-fine refinement.
Produces:
  - Bi-directional optical flow (t₀→t, t₁→t)
  - Backward-warped frames
  - Blend mask
  - Coarse interpolated frame

Architecture:
  IFNet (3-level pyramid) → Backward Warp → Mask-weighted Blend
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .warplayer import warp


def conv_block(in_ch, out_ch, stride=1):
    """Standard conv → PReLU block"""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1),
        nn.PReLU(out_ch),
    )


class IFBlock(nn.Module):
    """
    Single scale of the IFNet pyramid.
    Takes concatenated inputs and predicts:
      - flow_0→t (2 channels)
      - flow_1→t (2 channels)
      - blend mask (1 channel)
    Total output: 5 channels
    """
    def __init__(self, in_channels, hidden=64, out_channels=5):
        super().__init__()
        self.conv = nn.Sequential(
            conv_block(in_channels, hidden),
            conv_block(hidden, hidden),
            conv_block(hidden, hidden),
            conv_block(hidden, hidden),
            nn.Conv2d(hidden, out_channels, 3, padding=1),
        )

    def forward(self, x):
        return self.conv(x)


class IFNet(nn.Module):
    """
    Multi-scale optical flow estimation network.
    3-level coarse-to-fine pyramid:
      Level 0: 4x downsampled → coarsest flow
      Level 1: 2x downsampled → refined flow
      Level 2: 1x (full res) → final flow
    
    At each level, the flow from the previous level is upscaled
    and used to warp the inputs, so the network only needs to
    estimate the residual flow correction.
    """
    def __init__(self):
        super().__init__()
        # Level 0: Just the two images (2 channels each for grayscale, but we support 1-ch)
        self.block0 = IFBlock(in_channels=2, hidden=64, out_channels=5)
        # Level 1: images + upscaled predictions from level 0
        self.block1 = IFBlock(in_channels=2 + 5, hidden=64, out_channels=5)
        # Level 2: images + upscaled predictions from level 1
        self.block2 = IFBlock(in_channels=2 + 5, hidden=48, out_channels=5)

    def forward(self, img0, img1, t=0.5, scale_list=[4, 2, 1]):
        """
        Args:
            img0: (B, 1, H, W) frame at t=0
            img1: (B, 1, H, W) frame at t=1
            t: interpolation timestep (0.5 = midpoint)
            scale_list: downsampling factors for each pyramid level
        
        Returns:
            flow: (B, 4, H, W) — flow_0→t (2ch) + flow_1→t (2ch)
            mask: (B, 1, H, W) — blend mask (sigmoid)
            warped0: (B, 1, H, W) — img0 warped to time t
            warped1: (B, 1, H, W) — img1 warped to time t
        """
        B, C, H, W = img0.shape
        
        # Initialize flow and mask
        flow = torch.zeros(B, 4, H, W, device=img0.device, dtype=img0.dtype)
        mask = torch.zeros(B, 1, H, W, device=img0.device, dtype=img0.dtype)

        blocks = [self.block0, self.block1, self.block2]

        for i, (block, scale) in enumerate(zip(blocks, scale_list)):
            # Downsample images to current scale
            if scale != 1:
                img0_s = F.interpolate(img0, scale_factor=1.0/scale,
                                       mode='bilinear', align_corners=True)
                img1_s = F.interpolate(img1, scale_factor=1.0/scale,
                                       mode='bilinear', align_corners=True)
            else:
                img0_s = img0
                img1_s = img1

            if i == 0:
                # First level: just the two images
                x = torch.cat([img0_s, img1_s], dim=1)
            else:
                # Subsequent levels: images + upscaled flow/mask from previous level
                flow_s = F.interpolate(flow, size=img0_s.shape[2:],
                                       mode='bilinear', align_corners=True)
                # Scale flow values proportionally
                flow_s[:, :2] *= (img0_s.shape[3] / flow.shape[3])
                flow_s[:, 2:] *= (img0_s.shape[3] / flow.shape[3])
                mask_s = F.interpolate(mask, size=img0_s.shape[2:],
                                       mode='bilinear', align_corners=True)

                # Warp images using current flow estimate
                warped0_s = warp(img0_s, flow_s[:, :2] * t)
                warped1_s = warp(img1_s, flow_s[:, 2:4] * (1 - t))

                x = torch.cat([warped0_s, warped1_s, flow_s, mask_s], dim=1)

            # Predict flow residual + mask at this scale
            pred = block(x)
            delta_flow = pred[:, :4]  # flow_0→t (2) + flow_1→t (2)
            delta_mask = pred[:, 4:5]  # blend mask

            # Upsample to full resolution and accumulate
            if scale != 1:
                delta_flow = F.interpolate(delta_flow, size=(H, W),
                                           mode='bilinear', align_corners=True)
                delta_flow[:, :2] *= scale
                delta_flow[:, 2:] *= scale
                delta_mask = F.interpolate(delta_mask, size=(H, W),
                                           mode='bilinear', align_corners=True)

            flow = flow + delta_flow
            mask = mask + delta_mask

        # Apply sigmoid to mask
        mask = torch.sigmoid(mask)

        # Final warping with accumulated flow
        warped0 = warp(img0, flow[:, :2] * t)
        warped1 = warp(img1, flow[:, 2:4] * (1 - t))

        return flow, mask, warped0, warped1


class FlowInterpolator(nn.Module):
    """
    Model 1: Complete flow-based interpolator.
    
    Uses IFNet for optical flow estimation, then produces a coarse
    interpolated frame via mask-weighted blending of warped frames.
    """
    def __init__(self):
        super().__init__()
        self.ifnet = IFNet()

    def forward(self, img0, img1, t=0.5):
        """
        Args:
            img0: (B, 1, H, W) frame at time 0
            img1: (B, 1, H, W) frame at time 1
            t: interpolation time (0 = img0, 1 = img1, 0.5 = midpoint)
        
        Returns:
            dict with:
                coarse_frame: (B, 1, H, W) mask-weighted blend
                warped0: (B, 1, H, W) img0 warped to time t
                warped1: (B, 1, H, W) img1 warped to time t
                flow: (B, 4, H, W) optical flow fields
                mask: (B, 1, H, W) blending mask
        """
        flow, mask, warped0, warped1 = self.ifnet(img0, img1, t=t)

        # Coarse interpolation: mask-weighted blend
        coarse = mask * warped0 + (1 - mask) * warped1

        return {
            'coarse_frame': coarse,
            'warped0': warped0,
            'warped1': warped1,
            'flow': flow,
            'mask': mask,
        }

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
