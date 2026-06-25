"""
=================================================================
 ISRO Hackathon — 2-Model Satellite Frame Interpolation
 Google Colab Training Notebook (run as .py or paste cells in Colab)
=================================================================

Instructions:
  1. Upload this entire project folder to Google Drive
  2. Open a Colab notebook with GPU runtime (T4 or better)
  3. Copy-paste each section below as a separate cell
  4. Run sequentially

Quick test flow:
  Cell 1: Mount Drive + Install deps
  Cell 2: Quick sanity test with synthetic data
  Cell 3: Download real GOES data + preprocess
  Cell 4: Stage 1 training
  Cell 5: Stage 2 training  
  Cell 6: Stage 3 E2E fine-tuning
  Cell 7: Evaluate + visualize
"""

# ============================================================
# CELL 1: Setup Environment
# ============================================================
"""
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Set project path — UPDATE THIS to your Drive path
import os
PROJECT_DIR = '/content/drive/MyDrive/ISRO_Hackathon'  # <-- CHANGE THIS

# Clone or copy project files
# Option A: If you uploaded the zip
# !unzip -q "{PROJECT_DIR}/project.zip" -d /content/project
# Option B: If you uploaded the folder directly
!cp -r "{PROJECT_DIR}" /content/project

os.chdir('/content/project')

# Install dependencies
!pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu121
!pip install -q s3fs xarray netCDF4 h5py scikit-image piq lpips tensorboard tqdm matplotlib pandas

# Verify GPU
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
"""

# ============================================================
# CELL 2: Quick Sanity Test with Synthetic Data
# ============================================================
"""
import torch
import sys
sys.path.insert(0, '/content/project')

from models.pipeline import SatelliteInterpolator
from models.losses import TwoModelLoss
from dataset import SyntheticSatelliteDataset

device = torch.device('cuda')

# Create model
model = SatelliteInterpolator().to(device)
model.summary()

# Test forward pass
ds = SyntheticSatelliteDataset(num_samples=10, image_size=256)
f0, gt, f1 = ds[0]
f0 = f0.unsqueeze(0).to(device)
gt = gt.unsqueeze(0).to(device)
f1 = f1.unsqueeze(0).to(device)

# Forward pass
with torch.no_grad():
    out = model(f0, f1, t=0.5, refine=True)

print(f"\\nInput shape:    {f0.shape}")
print(f"Coarse shape:   {out['coarse'].shape}")
print(f"Refined shape:  {out['refined'].shape}")
print(f"Flow shape:     {out['flow'].shape}")
print(f"Residual range: [{out['residual'].min():.4f}, {out['residual'].max():.4f}]")

# Test loss
criterion = TwoModelLoss(stage=2).to(device)
loss, ld = criterion(out, gt)
print(f"\\nLoss: {loss.item():.4f}")
for k, v in ld.items():
    print(f"  {k}: {v:.4f}")

print("\\n✅ Sanity test passed!")
"""

# ============================================================
# CELL 3: Download and Preprocess Real Data (GOES-19)
# ============================================================
"""
# Download 24 hours of GOES data (~144 files, 10-min intervals)
!python scripts/download_goes19.py --date 2025-06-01 --hours 24 --output data/goes19/

# Preprocess to normalized .npy frames
!python scripts/preprocess.py --source goes19 --input data/goes19/ --output data/processed/frames/ --size 512

# Create train/val/test triplets
!python scripts/create_triplets.py --frames data/processed/frames/ --output data/processed/ --stride 2

# Check data
import os
for split in ['train', 'val', 'test']:
    d = f'data/processed/{split}'
    n = len(os.listdir(d)) - 1 if os.path.exists(d) else 0  # -1 for triplets.json
    print(f"{split}: {n} triplets")
"""

# ============================================================
# CELL 4: Stage 1 — Train Flow Model
# ============================================================
"""
# Train with synthetic data first (for testing), then real data
# For synthetic test:
!python train_stage1.py --synthetic --epochs 30 --batch_size 8 --crop_size 256 --save_dir checkpoints

# For real GOES-19 data:
# !python train_stage1.py --data_dir data/processed --epochs 100 --batch_size 8 --crop_size 256 --save_dir checkpoints

# Monitor with TensorBoard
%load_ext tensorboard
%tensorboard --logdir runs/stage1
"""

# ============================================================
# CELL 5: Stage 2 — Train Refinement Model (Frozen Flow)
# ============================================================
"""
# Synthetic:
!python train_stage2.py --synthetic --resume checkpoints/flow_model_best.pth --epochs 30 --batch_size 4

# Real data:
# !python train_stage2.py --data_dir data/processed --resume checkpoints/flow_model_best.pth --epochs 80 --batch_size 4

%tensorboard --logdir runs/stage2
"""

# ============================================================
# CELL 6: Stage 3 — End-to-End Fine-tuning
# ============================================================
"""
# Synthetic:
!python train_e2e.py --synthetic --resume checkpoints/refinement_model_best.pth --epochs 15 --batch_size 4

# Real data:
# !python train_e2e.py --data_dir data/processed --resume checkpoints/refinement_model_best.pth --epochs 30 --batch_size 4

%tensorboard --logdir runs/stage3_e2e
"""

# ============================================================
# CELL 7: Visualize Results
# ============================================================
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from models.pipeline import SatelliteInterpolator
from dataset import SyntheticSatelliteDataset

device = torch.device('cuda')
model = SatelliteInterpolator().to(device)

# Load best checkpoint
ckpt = torch.load('checkpoints/e2e_model_best.pth', map_location=device)
model.load_state_dict(ckpt['model_state'])
model.eval()

# Get a test sample
ds = SyntheticSatelliteDataset(100, 256)
f0, gt, f1 = ds[0]
f0 = f0.unsqueeze(0).to(device)
gt_tensor = gt.unsqueeze(0).to(device)
f1 = f1.unsqueeze(0).to(device)

with torch.no_grad():
    out = model(f0, f1, t=0.5, refine=True)

# Visualize
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
images = [
    ('Frame t₀', f0),
    ('Ground Truth', gt_tensor),
    ('Frame t₁', f1),
    ('Coarse (Model 1)', out['coarse']),
    ('Refined (Model 1+2)', out['refined']),
    ('Residual (×10)', out['residual'] * 10 + 0.5),
    ('Blend Mask', out['mask']),
    ('|Error| (×5)', (torch.abs(out['refined'] - gt_tensor) * 5)),
]
for ax, (title, img) in zip(axes.flat, images):
    ax.imshow(img.squeeze().cpu().numpy(), cmap='inferno', vmin=0, vmax=1)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.axis('off')
plt.suptitle('2-Model Satellite Frame Interpolation Results', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('output/visualization.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved to output/visualization.png")
"""

# ============================================================
# CELL 8: Recursive Interpolation Demo
# ============================================================
"""
import torch
import matplotlib.pyplot as plt
from models.pipeline import SatelliteInterpolator
from dataset import SyntheticSatelliteDataset

device = torch.device('cuda')
model = SatelliteInterpolator().to(device)
ckpt = torch.load('checkpoints/e2e_model_best.pth', map_location=device)
model.load_state_dict(ckpt['model_state'])
model.eval()

ds = SyntheticSatelliteDataset(10, 256)
f0, gt, f1 = ds[0]
f0 = f0.unsqueeze(0).to(device)
f1 = f1.unsqueeze(0).to(device)

# Recursive: depth=2 → 3 intermediate frames (4x temporal resolution)
with torch.no_grad():
    interpolated = model.recursive_interpolate(f0, f1, depth=2)

print(f"Input: 2 frames → Output: {len(interpolated)} intermediate frames")
print(f"Temporal resolution: 30min → {30 / (len(interpolated) + 1):.1f}min")

# Visualize full sequence
all_frames = [f0] + interpolated + [f1]
fig, axes = plt.subplots(1, len(all_frames), figsize=(4 * len(all_frames), 4))
labels = ['t=0 (orig)']
for i in range(len(interpolated)):
    t = (i + 1) / (len(interpolated) + 1)
    labels.append(f't={t:.2f} (interp)')
labels.append('t=1 (orig)')

for ax, frame, label in zip(axes, all_frames, labels):
    ax.imshow(frame.squeeze().cpu().numpy(), cmap='inferno')
    color = 'green' if 'orig' in label else 'cyan'
    ax.set_title(label, fontsize=10, color=color, fontweight='bold')
    ax.axis('off')
plt.suptitle('Recursive Interpolation (depth=2)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('output/recursive_demo.png', dpi=150, bbox_inches='tight')
plt.show()
"""

# ============================================================
# CELL 9: Save model to Google Drive
# ============================================================
"""
import shutil
SAVE_DIR = '/content/drive/MyDrive/ISRO_Hackathon/trained_models/'
os.makedirs(SAVE_DIR, exist_ok=True)
for f in ['flow_model_best.pth', 'refinement_model_best.pth', 'e2e_model_best.pth']:
    src = f'checkpoints/{f}'
    if os.path.exists(src):
        shutil.copy2(src, os.path.join(SAVE_DIR, f))
        print(f"Saved {f} to Drive")
print("\\n✅ Models saved to Google Drive!")
"""
