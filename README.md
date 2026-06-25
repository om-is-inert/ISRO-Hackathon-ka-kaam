# ISRO Hackathon: 2-Model Satellite Frame Interpolation

AI/ML pipeline for enhancing temporal resolution of geostationary satellite imagery using a **2-model approach**: Flow-based Interpolation + Attention-based Refinement.

## Architecture

```
Frame t₀ ──┐                                        
            ├──→ [Model 1: IFNet]  ──→ Coarse Frame ──→ [Model 2: RefinementNet] ──→ Final Frame
Frame t₁ ──┘    (Optical Flow)         + Warps/Flow     (U-Net + CBAM Attention)
```

**Model 1 (FlowInterpolator):** Multi-scale optical flow estimation → backward warping → mask-weighted blend (~10M params)

**Model 2 (RefinementNet):** U-Net with CBAM attention that learns residual corrections over Model 1's coarse output (~5-8M params)

## Quick Start (Google Colab)

1. Upload project to Google Drive
2. Open `colab_notebook.py` and copy cells into a Colab notebook
3. Select **GPU runtime** (T4 or better)
4. Run cells sequentially

### Test with synthetic data (no download needed):
```bash
python train_stage1.py --synthetic --epochs 30 --batch_size 8
python train_stage2.py --synthetic --resume checkpoints/flow_model_best.pth --epochs 30
python train_e2e.py --synthetic --resume checkpoints/refinement_model_best.pth --epochs 15
```

### Full pipeline with GOES-19 data:
```bash
# 1. Download
python scripts/download_goes19.py --date 2025-06-01 --hours 24

# 2. Preprocess
python scripts/preprocess.py --source goes19 --input data/goes19/ --output data/processed/frames/

# 3. Create triplets
python scripts/create_triplets.py --frames data/processed/frames/ --output data/processed/

# 4. Train (3 stages)
python train_stage1.py --data_dir data/processed --epochs 100
python train_stage2.py --data_dir data/processed --resume checkpoints/flow_model_best.pth --epochs 80
python train_e2e.py --data_dir data/processed --resume checkpoints/refinement_model_best.pth --epochs 30

# 5. Evaluate
python evaluate.py --checkpoint checkpoints/e2e_model_best.pth --data_dir data/processed
```

## Project Structure

```
├── models/
│   ├── flow_model.py          # Model 1: IFNet optical flow estimator
│   ├── refinement_model.py    # Model 2: U-Net + CBAM refinement
│   ├── pipeline.py            # Combined SatelliteInterpolator
│   ├── losses.py              # SSIM, perceptual, edge, flow losses
│   └── warplayer.py           # Differentiable backward warping
├── scripts/
│   ├── download_goes19.py     # AWS S3 data download
│   ├── preprocess.py          # NC/H5 → normalized .npy
│   └── create_triplets.py     # Train/val/test split
├── dataset.py                 # PyTorch datasets
├── train_stage1.py            # Stage 1: Flow model only
├── train_stage2.py            # Stage 2: Refinement (frozen flow)
├── train_stage3.py            # Stage 3: End-to-end fine-tuning
├── inference.py               # Single + recursive interpolation
├── evaluate.py                # PSNR/SSIM/MSE metrics
├── colab_notebook.py          # Google Colab instructions
└── requirements.txt
```

## 3-Stage Training

| Stage | What | Freeze | LR | Epochs |
|-------|------|--------|----|--------|
| 1 | Train Flow Model | — | 1e-4 | 100 |
| 2 | Train Refinement | Flow Model | 2e-4 | 80 |
| 3 | End-to-End | — | 1e-5 | 30 |

## Data Sources

- **GOES-19/16** (training): AWS S3 public, 10-min cadence, Ch13 TIR
- **INSAT-3DS** (application): MOSDAC, 30-min cadence, TIR1 band
