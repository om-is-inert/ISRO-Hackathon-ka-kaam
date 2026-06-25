"""
Differentiable backward warping module.
Given an image and an optical flow field, warps the image according to the flow
using bilinear interpolation (fully differentiable for backprop).
"""

import torch
import torch.nn.functional as F


def warp(img, flow):
    """
    Backward-warp an image using optical flow.
    
    Args:
        img: (B, C, H, W) input image tensor
        flow: (B, 2, H, W) optical flow field (dx, dy)
    
    Returns:
        warped: (B, C, H, W) warped image
    """
    B, C, H, W = img.shape

    # Create meshgrid of pixel coordinates
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=img.device, dtype=img.dtype),
        torch.arange(W, device=img.device, dtype=img.dtype),
        indexing='ij'
    )

    # Add flow offsets to get sampling locations
    grid_x = grid_x.unsqueeze(0).expand(B, -1, -1) + flow[:, 0, :, :]
    grid_y = grid_y.unsqueeze(0).expand(B, -1, -1) + flow[:, 1, :, :]

    # Normalize to [-1, 1] for grid_sample
    grid_x = 2.0 * grid_x / (W - 1) - 1.0
    grid_y = 2.0 * grid_y / (H - 1) - 1.0

    # Stack into sampling grid (B, H, W, 2)
    grid = torch.stack([grid_x, grid_y], dim=-1)

    # Bilinear sampling (differentiable)
    warped = F.grid_sample(
        img,
        grid,
        mode='bilinear',
        padding_mode='border',
        align_corners=True
    )

    return warped


def multi_scale_warp(img, flow, scale=1.0):
    """
    Warp at a specific scale — useful for coarse-to-fine flow estimation.
    
    Args:
        img: (B, C, H, W) image
        flow: (B, 2, H', W') flow at potentially different resolution
        scale: flow magnitude scale factor
    
    Returns:
        warped: (B, C, H, W) warped image at original resolution
    """
    B, C, H, W = img.shape
    _, _, fH, fW = flow.shape

    # Resize flow to image resolution if needed
    if fH != H or fW != W:
        flow = F.interpolate(flow, size=(H, W), mode='bilinear', align_corners=True)
        # Scale flow values proportionally
        flow[:, 0] *= (W / fW)
        flow[:, 1] *= (H / fH)

    flow = flow * scale
    return warp(img, flow)
