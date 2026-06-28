"""
Unified Training Script for 2-Model Satellite Frame Interpolation Pipeline.

Supports Stage 1 (Optical Flow), Stage 2 (Refinement), and Stage 3 (End-to-End Finetuning)
via the `--stage` argument. Includes automatic mixed precision (AMP), gradient accumulation,
and dynamic CPU worker allocation.

Usage:
    # Stage 1: Train Optical Flow
    python train.py --stage 1 --data_dir data/processed --epochs 100 --batch_size 8 --use_amp

    # Stage 2: Train Refinement Model
    python train.py --stage 2 --data_dir data/processed --resume checkpoints/flow_model_best.pth --epochs 50 --batch_size 8 --use_amp

    # Stage 3: End-to-End Finetuning
    python train.py --stage 3 --data_dir data/processed --resume checkpoints/refinement_model_best.pth --epochs 50 --batch_size 4 --use_amp
"""
import os
import time
import argparse
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.pipeline import SatelliteInterpolator
from models.losses import TwoModelLoss
from dataset import SatelliteTripletDataset, SyntheticSatelliteDataset


def get_default_workers():
    """Dynamically determine optimal dataloader workers to avoid CPU thread starvation."""
    return min(4, os.cpu_count() or 1)


def parse_args():
    p = argparse.ArgumentParser(description='Unified Training Script for Satellite Interpolation')
    p.add_argument('--stage', type=int, choices=[1, 2, 3], required=True,
                   help='Training stage: 1=Flow Model, 2=Refinement Model, 3=End-to-End')
    p.add_argument('--data_dir', type=str, default='data/processed', help='Path to processed dataset')
    p.add_argument('--synthetic', action='store_true', help='Use synthetic data for testing pipeline')
    p.add_argument('--resume', type=str, default=None, help='Path to initial or resumed checkpoint')
    p.add_argument('--epochs', type=int, default=None, help='Number of epochs (defaults to stage-specific setting)')
    p.add_argument('--batch_size', type=int, default=None, help='Batch size (defaults to stage-specific setting)')
    p.add_argument('--lr', type=float, default=None, help='Learning rate')
    p.add_argument('--crop_size', type=int, default=256, help='Spatial training crop size')
    p.add_argument('--num_workers', type=int, default=get_default_workers(), help='DataLoader CPU workers')
    p.add_argument('--accum_steps', type=int, default=2, help='Gradient accumulation steps')
    p.add_argument('--use_amp', action='store_true', help='Use automatic mixed precision (FP16)')
    p.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    p.add_argument('--log_dir', type=str, default=None, help='TensorBoard log directory')
    p.add_argument('--val_every', type=int, default=5, help='Validate every N epochs')
    return p.parse_args()


def validate(model, val_loader, criterion, device, refine=False):
    model.eval()
    total_loss, total_ssim, n = 0.0, 0.0, 0
    ssim_key = 'refined_ssim' if refine else 'coarse_ssim'
    
    with torch.no_grad():
        for f0, gt, f1 in val_loader:
            f0, gt, f1 = f0.to(device), gt.to(device), f1.to(device)
            out = model(f0, f1, t=0.5, refine=refine)
            loss, ld = criterion(out, gt)
            
            batch_size = f0.size(0)
            total_loss += loss.item() * batch_size
            total_ssim += (1.0 - ld.get(ssim_key, 0.0)) * batch_size
            n += batch_size
            
    return total_loss / max(n, 1), total_ssim / max(n, 1)


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Configure stage-specific defaults
    if args.stage == 1:
        epochs = args.epochs if args.epochs is not None else 100
        batch_size = args.batch_size if args.batch_size is not None else 8
        lr = args.lr if args.lr is not None else 1e-4
        log_dir = args.log_dir or 'runs/stage1_flow'
        ckpt_name = 'flow_model_best.pth'
        refine = False
    elif args.stage == 2:
        if not args.resume and not args.synthetic:
            raise ValueError("Stage 2 requires --resume pointing to a trained Stage 1 flow checkpoint!")
        epochs = args.epochs if args.epochs is not None else 50
        batch_size = args.batch_size if args.batch_size is not None else 8
        lr = args.lr if args.lr is not None else 2e-4
        log_dir = args.log_dir or 'runs/stage2_refinement'
        ckpt_name = 'refinement_model_best.pth'
        refine = True
    elif args.stage == 3:
        if not args.resume and not args.synthetic:
            raise ValueError("Stage 3 requires --resume pointing to a trained Stage 2 refinement checkpoint!")
        epochs = args.epochs if args.epochs is not None else 50
        batch_size = args.batch_size if args.batch_size is not None else 4
        lr = args.lr if args.lr is not None else 1e-5
        log_dir = args.log_dir or 'runs/stage3_e2e'
        ckpt_name = 'e2e_model_best.pth'
        refine = True

    print(f"=== Unified Trainer | Stage {args.stage} ===")
    print(f"Device: {device} | Epochs: {epochs} | Batch Size: {batch_size} | LR: {lr:.2e}")
    print(f"Workers: {args.num_workers} | Accum Steps: {args.accum_steps} | AMP: {'ON' if args.use_amp else 'OFF'}")

    # Datasets
    if args.synthetic:
        train_ds = SyntheticSatelliteDataset(num_samples=100, image_size=args.crop_size)
        val_ds = SyntheticSatelliteDataset(num_samples=20, image_size=args.crop_size)
    else:
        train_ds = SatelliteTripletDataset(args.data_dir, split='train', crop_size=args.crop_size, augment=True)
        val_ds = SatelliteTripletDataset(args.data_dir, split='val', crop_size=args.crop_size, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=(device.type == 'cuda'))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=(device.type == 'cuda'))

    # Model & Loss
    model = SatelliteInterpolator().to(device)
    criterion = TwoModelLoss(stage=args.stage).to(device)

    # Resume Checkpoint Loading
    start_epoch = 0
    best_val_loss = float('inf')
    if args.resume and os.path.exists(args.resume):
        print(f"Loading checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        state_dict = ckpt['model_state'] if 'model_state' in ckpt else ckpt
        model.load_state_dict(state_dict, strict=False)
        if args.stage == 1 and 'epoch' in ckpt:
            start_epoch = ckpt['epoch'] + 1
            best_val_loss = ckpt.get('best_val_loss', float('inf'))

    # Optimizers & Parameter Clipping Setup
    if args.stage == 1:
        optimizer = torch.optim.AdamW(model.flow_model.parameters(), lr=lr, weight_decay=1e-4)
        clip_params = model.flow_model.parameters()
        max_norm = 1.0
    elif args.stage == 2:
        optimizer = torch.optim.AdamW(model.refinement_model.parameters(), lr=lr, weight_decay=1e-4)
        clip_params = model.refinement_model.parameters()
        max_norm = 1.0
    elif args.stage == 3:
        # Differential learning rates for E2E
        optimizer = torch.optim.AdamW([
            {'params': model.flow_model.parameters(), 'lr': lr * 0.1},
            {'params': model.refinement_model.parameters(), 'lr': lr},
        ], weight_decay=1e-4)
        clip_params = model.parameters()
        max_norm = 0.5

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 1e-2)
    writer = SummaryWriter(log_dir)

    use_amp = args.use_amp and device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    accum_steps = max(1, args.accum_steps)

    # Training Loop
    for epoch in range(start_epoch, epochs):
        model.train()
        if args.stage == 2:
            model.flow_model.eval()  # Keep flow weights frozen/eval mode in Stage 2

        epoch_loss, batch_count = 0.0, 0
        t0 = time.time()
        optimizer.zero_grad()

        for i, (f0, gt, f1) in enumerate(train_loader):
            f0, gt, f1 = f0.to(device), gt.to(device), f1.to(device)

            with torch.cuda.amp.autocast(enabled=use_amp):
                out = model(f0, f1, t=0.5, refine=refine)
                loss, ld = criterion(out, gt)
                loss_scaled = loss / accum_steps

            scaler.scale(loss_scaled).backward()
            epoch_loss += loss.item() * accum_steps
            batch_count += 1

            if (i + 1) % accum_steps == 0 or (i + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(clip_params, max_norm=max_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            if i % 20 == 0:
                step = epoch * len(train_loader) + i
                for k, v in ld.items():
                    writer.add_scalar(f'train/{k}', v, step)
                ssim_val = 1.0 - ld.get('refined_ssim' if refine else 'coarse_ssim', 0.0)
                print(f"  [{epoch}/{epochs}] batch {i}/{len(train_loader)} | Loss: {loss.item()*accum_steps:.4f} | SSIM: {ssim_val:.4f}")

        scheduler.step()
        avg_loss = epoch_loss / max(batch_count, 1)
        elapsed = time.time() - t0
        writer.add_scalar('train/epoch_loss', avg_loss, epoch)
        writer.add_scalar('train/lr', scheduler.get_last_lr()[0], epoch)
        print(f"Epoch {epoch} complete | Avg Loss: {avg_loss:.4f} | Time: {elapsed:.1f}s")

        # Validation & Checkpointing
        if epoch % args.val_every == 0 or epoch == epochs - 1:
            val_loss, val_ssim = validate(model, val_loader, criterion, device, refine=refine)
            writer.add_scalar('val/loss', val_loss, epoch)
            writer.add_scalar('val/ssim', val_ssim, epoch)
            print(f"  --> VAL Loss: {val_loss:.4f} | VAL SSIM: {val_ssim:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = os.path.join(args.save_dir, ckpt_name)
                torch.save({
                    'epoch': epoch,
                    'stage': args.stage,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                }, save_path)
                print(f"  [★] Saved new best model to {save_path} (Val Loss: {best_val_loss:.4f})")

    writer.close()
    print(f"\nStage {args.stage} Training Complete! Best Val Loss: {best_val_loss:.4f}")
    print(f"Best model saved at: {os.path.join(args.save_dir, ckpt_name)}")


if __name__ == '__main__':
    main()
