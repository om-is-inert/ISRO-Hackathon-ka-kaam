"""
Combined 2-Model Pipeline: SatelliteInterpolator

Chains Model 1 (FlowInterpolator) → Model 2 (RefinementNet)
into a single forward pass. Supports:
  - Full pipeline inference (flow → refine)
  - Coarse-only mode (Model 1 only, for Stage 1 training)
  - Recursive interpolation (30min → 15min → 7.5min)
  - Mixed-precision inference
  - Gradient checkpointing toggle for VRAM optimization
"""

import torch
import torch.nn as nn

from .flow_model import FlowInterpolator
from .refinement_model import RefinementNet


class SatelliteInterpolator(nn.Module):
    """
    Complete 2-model satellite frame interpolation pipeline.
    
    Stage 1: FlowInterpolator estimates optical flow and produces
             a coarse interpolated frame via backward warping.
    Stage 2: RefinementNet takes the coarse output + all intermediates
             and learns a residual correction for sharp, clean output.
    
    Optimized for free-tier GPU:
      - base_channels=32 (was 48) — ~55% fewer refinement params
      - Gradient checkpointing toggle for training VRAM savings
      - AMP-compatible forward pass
    """
    def __init__(self, refine_in_channels=8, refine_base_channels=32):
        super().__init__()
        self.flow_model = FlowInterpolator()
        self.refinement_model = RefinementNet(
            in_channels=refine_in_channels,
            base_channels=refine_base_channels,
        )
    
    def enable_checkpointing(self):
        """Enable gradient checkpointing on refinement model bottleneck."""
        self.refinement_model.use_checkpointing = True
        print("[Checkpointing] Enabled on refinement bottleneck — saves ~30% VRAM")
    
    def disable_checkpointing(self):
        """Disable gradient checkpointing (faster inference)."""
        self.refinement_model.use_checkpointing = False
    
    def forward(self, img0, img1, t=0.5, refine=True):
        """
        Args:
            img0: (B, 1, H, W) frame at time 0
            img1: (B, 1, H, W) frame at time 1
            t: interpolation timestep (0.5 = midpoint)
            refine: if False, skip refinement (for Stage 1 training)
        
        Returns:
            dict with coarse, refined, residual, flow, mask
        """
        # Stage 1: Flow-based coarse interpolation
        stage1 = self.flow_model(img0, img1, t=t)
        
        if not refine:
            return {
                'coarse': stage1['coarse_frame'],
                'refined': stage1['coarse_frame'],  # No refinement
                'residual': torch.zeros_like(stage1['coarse_frame']),
                'flow': stage1['flow'],
                'mask': stage1['mask'],
                'warped0': stage1['warped0'],
                'warped1': stage1['warped1'],
            }
        
        # Stage 2: Refinement
        refined, residual = self.refinement_model(
            coarse_frame=stage1['coarse_frame'],
            warped0=stage1['warped0'],
            warped1=stage1['warped1'],
            img0=img0,
            img1=img1,
            flow=stage1['flow'],
            mask=stage1['mask'],
        )
        
        return {
            'coarse': stage1['coarse_frame'],
            'refined': refined,
            'residual': residual,
            'flow': stage1['flow'],
            'mask': stage1['mask'],
            'warped0': stage1['warped0'],
            'warped1': stage1['warped1'],
        }
    
    @torch.no_grad()
    def inference(self, img0, img1, t=0.5):
        """Single-frame inference (no grad, for evaluation/deployment)."""
        self.eval()
        outputs = self.forward(img0, img1, t=t, refine=True)
        return outputs['refined']
    
    @torch.no_grad()
    def recursive_interpolate(self, img0, img1, depth=2):
        """
        Recursively interpolate to increase temporal resolution.
        
        depth=1: 1 intermediate frame  (30min → 15min)
        depth=2: 3 intermediate frames (30min → 7.5min)
        depth=3: 7 intermediate frames (30min → 3.75min)
        
        Returns:
            list of interpolated frames (excluding img0 and img1)
        """
        self.eval()
        if depth == 0:
            return []
        
        # Interpolate midpoint
        mid = self.inference(img0, img1, t=0.5)
        
        # Recurse on left and right halves
        left = self.recursive_interpolate(img0, mid, depth - 1)
        right = self.recursive_interpolate(mid, img1, depth - 1)
        
        return left + [mid] + right
    
    def freeze_flow_model(self):
        """Freeze Model 1 for Stage 2 training."""
        for param in self.flow_model.parameters():
            param.requires_grad = False
        print(f"[Frozen] Flow model — {self.flow_model.count_parameters():,} params frozen")
    
    def unfreeze_flow_model(self):
        """Unfreeze Model 1 for end-to-end fine-tuning."""
        for param in self.flow_model.parameters():
            param.requires_grad = True
        print(f"[Unfrozen] Flow model — {self.flow_model.count_parameters():,} params active")
    
    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total': total, 'trainable': trainable}
    
    def summary(self):
        flow_params = self.flow_model.count_parameters()
        refine_params = self.refinement_model.count_parameters()
        counts = self.count_parameters()
        print(f"╔══════════════════════════════════════════╗")
        print(f"║   Satellite Interpolator — 2 Model       ║")
        print(f"╠══════════════════════════════════════════╣")
        print(f"║ Model 1 (Flow):       {flow_params:>12,} params ║")
        print(f"║ Model 2 (Refinement): {refine_params:>12,} params ║")
        print(f"╠══════════════════════════════════════════╣")
        print(f"║ Total:                {counts['total']:>12,} params ║")
        print(f"║ Trainable:            {counts['trainable']:>12,} params ║")
        print(f"╚══════════════════════════════════════════╝")
