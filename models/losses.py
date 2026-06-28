"""
Loss functions for the 2-model satellite frame interpolation pipeline.

Includes:
  - L1 reconstruction loss
  - SSIM loss (structural similarity)
  - Perceptual loss (VGG feature matching — adapted for grayscale)
  - Edge-aware loss (Sobel filter)
  - Flow smoothness loss (total variation)
  - Residual regularization
  - Combined TwoModelLoss with configurable weights

Optimized for free-tier GPU:
  - VGG perceptual loss uses relu2_2 (layer 9) instead of relu3_3 (layer 16)
    → ~40% less VRAM for feature extraction
  - Lazy VGG loading: only initialized when stage >= 2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SSIMLoss(nn.Module):
    """Structural Similarity Index loss (1 - SSIM)."""
    def __init__(self, window_size=11, channel=1):
        super().__init__()
        self.window_size = window_size
        self.channel = channel
        
        # Create 1D Gaussian window
        sigma = 1.5
        gauss = torch.arange(window_size, dtype=torch.float32)
        gauss = gauss - window_size // 2
        gauss = torch.exp(-gauss.pow(2) / (2 * sigma ** 2))
        gauss = gauss / gauss.sum()
        
        # Create 2D window
        window = gauss.unsqueeze(1) * gauss.unsqueeze(0)
        window = window.unsqueeze(0).unsqueeze(0)
        self.register_buffer('window', window.expand(channel, 1, -1, -1).contiguous())
    
    def forward(self, pred, target):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        pad = self.window_size // 2
        
        mu1 = F.conv2d(pred, self.window, padding=pad, groups=self.channel)
        mu2 = F.conv2d(target, self.window, padding=pad, groups=self.channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(pred * pred, self.window, padding=pad, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(target * target, self.window, padding=pad, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(pred * target, self.window, padding=pad, groups=self.channel) - mu1_mu2
        
        ssim = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return 1.0 - ssim.mean()


class PerceptualLoss(nn.Module):
    """
    VGG-based perceptual loss (VRAM-optimized for free-tier GPU).
    Uses early VGG16 features up to relu2_2 (layer 9) instead of relu3_3 (layer 16).
    This saves ~40% GPU memory while still capturing texture/structure features.
    Adapted for grayscale: repeats single channel to 3 channels.
    """
    def __init__(self):
        super().__init__()
        import torchvision.models as models
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        # Use features up to relu2_2 (layer 9) — lighter than relu3_3 (layer 16)
        self.features = nn.Sequential(*list(vgg.features[:9]))
        # Freeze VGG — no gradients needed
        for param in self.features.parameters():
            param.requires_grad = False
        self.features.eval()  # Always in eval mode
        
        # ImageNet normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def _preprocess(self, x):
        """Convert grayscale to 3-channel and normalize."""
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return (x - self.mean) / self.std
    
    def forward(self, pred, target):
        # Run VGG in eval mode always, with no_grad for target features
        self.features.eval()
        pred_feat = self.features(self._preprocess(pred))
        with torch.no_grad():
            target_feat = self.features(self._preprocess(target))
        return F.l1_loss(pred_feat, target_feat)


class EdgeLoss(nn.Module):
    """Edge-aware loss using Sobel filter — preserves cloud boundaries."""
    def __init__(self):
        super().__init__()
        # Sobel kernels
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                               dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                               dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
    
    def _edges(self, x):
        gx = F.conv2d(x, self.sobel_x, padding=1)
        gy = F.conv2d(x, self.sobel_y, padding=1)
        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)
    
    def forward(self, pred, target):
        return F.l1_loss(self._edges(pred), self._edges(target))


class FlowSmoothnessLoss(nn.Module):
    """Total variation regularization on optical flow."""
    def forward(self, flow):
        dx = torch.abs(flow[:, :, :, 1:] - flow[:, :, :, :-1])
        dy = torch.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :])
        return dx.mean() + dy.mean()


class TwoModelLoss(nn.Module):
    """
    Combined loss for the 2-model pipeline.
    
    Supports three training stages:
      Stage 1: Only coarse loss (training flow model)
      Stage 2: Coarse + refined + perceptual + edge (training refinement)
      Stage 3: All losses end-to-end
    
    Args:
        stage: 1, 2, or 3
        lambda_coarse: weight for coarse reconstruction loss
        lambda_refined: weight for refined reconstruction loss
        lambda_perceptual: weight for VGG perceptual loss
        lambda_edge: weight for edge-aware loss
        lambda_flow_smooth: weight for flow smoothness
        lambda_residual_reg: weight for residual regularization
    """
    def __init__(self, stage=1,
                 lambda_coarse=1.0,
                 lambda_refined=1.0,
                 lambda_perceptual=0.1,
                 lambda_edge=0.5,
                 lambda_flow_smooth=0.01,
                 lambda_residual_reg=0.01):
        super().__init__()
        self.stage = stage
        
        # Loss components — always needed
        self.l1 = nn.L1Loss()
        self.ssim = SSIMLoss(channel=1)
        self.flow_smooth = FlowSmoothnessLoss()
        
        # Stage 2+ losses — only load VGG when needed (saves VRAM in Stage 1)
        if stage >= 2:
            self.perceptual = PerceptualLoss()
            self.edge = EdgeLoss()
        else:
            self.perceptual = None
            self.edge = None
        
        # Weights
        self.w = {
            'coarse': lambda_coarse,
            'refined': lambda_refined,
            'perceptual': lambda_perceptual,
            'edge': lambda_edge,
            'flow_smooth': lambda_flow_smooth,
            'residual_reg': lambda_residual_reg,
        }
    
    def forward(self, outputs, gt):
        """
        Args:
            outputs: dict from SatelliteInterpolator.forward()
            gt: (B, 1, H, W) ground truth frame
        
        Returns:
            total_loss: scalar tensor
            loss_dict: dict of individual loss values (for logging)
        """
        coarse = outputs['coarse']
        refined = outputs['refined']
        residual = outputs['residual']
        flow = outputs['flow']
        
        loss_dict = {}
        total = 0.0
        
        # --- Coarse loss (always active) ---
        loss_coarse_l1 = self.l1(coarse, gt)
        loss_coarse_ssim = self.ssim(coarse, gt)
        loss_coarse = loss_coarse_l1 + loss_coarse_ssim
        loss_dict['coarse_l1'] = loss_coarse_l1.item()
        loss_dict['coarse_ssim'] = loss_coarse_ssim.item()
        total += self.w['coarse'] * loss_coarse
        
        if self.stage >= 2:
            # --- Refined loss ---
            loss_refined_l1 = self.l1(refined, gt)
            loss_refined_ssim = self.ssim(refined, gt)
            loss_refined = loss_refined_l1 + loss_refined_ssim
            loss_dict['refined_l1'] = loss_refined_l1.item()
            loss_dict['refined_ssim'] = loss_refined_ssim.item()
            total += self.w['refined'] * loss_refined
            
            # --- Perceptual loss ---
            loss_perceptual = self.perceptual(refined, gt)
            loss_dict['perceptual'] = loss_perceptual.item()
            total += self.w['perceptual'] * loss_perceptual
            
            # --- Edge loss ---
            loss_edge = self.edge(refined, gt)
            loss_dict['edge'] = loss_edge.item()
            total += self.w['edge'] * loss_edge
            
            # --- Residual regularization ---
            loss_reg = torch.mean(torch.abs(residual))
            loss_dict['residual_reg'] = loss_reg.item()
            total += self.w['residual_reg'] * loss_reg
        
        # --- Flow smoothness (always useful) ---
        loss_flow = self.flow_smooth(flow)
        loss_dict['flow_smooth'] = loss_flow.item()
        total += self.w['flow_smooth'] * loss_flow
        
        loss_dict['total'] = total.item()
        
        return total, loss_dict
