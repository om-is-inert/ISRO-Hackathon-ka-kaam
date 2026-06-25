"""
Stage 1: Train the Flow Model (Model 1) only.
Uses coarse reconstruction loss (L1 + SSIM + flow smoothness).
Can use synthetic data for initial testing, then GOES-19 data.
"""
import os, sys, argparse, time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from models.pipeline import SatelliteInterpolator
from models.losses import TwoModelLoss
from dataset import SatelliteTripletDataset, SyntheticSatelliteDataset

def parse_args():
    p = argparse.ArgumentParser(description='Stage 1: Train Flow Model')
    p.add_argument('--data_dir', type=str, default='data/processed')
    p.add_argument('--synthetic', action='store_true', help='Use synthetic data for testing')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--lr_min', type=float, default=1e-6)
    p.add_argument('--crop_size', type=int, default=256)
    p.add_argument('--num_workers', type=int, default=2)
    p.add_argument('--save_dir', type=str, default='checkpoints')
    p.add_argument('--log_dir', type=str, default='runs/stage1')
    p.add_argument('--save_every', type=int, default=10)
    p.add_argument('--val_every', type=int, default=5)
    p.add_argument('--resume', type=str, default=None, help='Path to checkpoint')
    return p.parse_args()

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss, total_ssim, n = 0, 0, 0
    with torch.no_grad():
        for f0, gt, f1 in val_loader:
            f0, gt, f1 = f0.to(device), gt.to(device), f1.to(device)
            out = model(f0, f1, t=0.5, refine=False)
            loss, ld = criterion(out, gt)
            total_loss += loss.item() * f0.size(0)
            total_ssim += (1 - ld.get('coarse_ssim', 0)) * f0.size(0)
            n += f0.size(0)
    return total_loss / n, total_ssim / n

def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Dataset
    if args.synthetic:
        print("Using SYNTHETIC data for testing pipeline")
        train_ds = SyntheticSatelliteDataset(num_samples=2000, image_size=args.crop_size)
        val_ds = SyntheticSatelliteDataset(num_samples=200, image_size=args.crop_size)
    else:
        train_ds = SatelliteTripletDataset(args.data_dir, split='train', crop_size=args.crop_size)
        val_ds = SatelliteTripletDataset(args.data_dir, split='val', crop_size=None, augment=False)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    print(f"Train: {len(train_ds)} samples | Val: {len(val_ds)} samples")

    # Model
    model = SatelliteInterpolator().to(device)
    model.summary()

    # Loss (Stage 1 — coarse only)
    criterion = TwoModelLoss(stage=1).to(device)

    # Optimizer + Scheduler
    optimizer = torch.optim.AdamW(model.flow_model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr_min)

    # Resume
    start_epoch = 0
    best_val_loss = float('inf')
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state'], strict=False)
        optimizer.load_state_dict(ckpt['optimizer_state'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        print(f"Resumed from epoch {start_epoch}")

    writer = SummaryWriter(args.log_dir)

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss, batch_count = 0, 0
        t0 = time.time()

        for i, (f0, gt, f1) in enumerate(train_loader):
            f0, gt, f1 = f0.to(device), gt.to(device), f1.to(device)
            out = model(f0, f1, t=0.5, refine=False)  # Stage 1: no refinement
            loss, ld = criterion(out, gt)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.flow_model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1

            if i % 20 == 0:
                step = epoch * len(train_loader) + i
                for k, v in ld.items():
                    writer.add_scalar(f'train/{k}', v, step)
                print(f"  [{epoch}/{args.epochs}] batch {i}/{len(train_loader)} "
                      f"loss={loss.item():.4f} coarse_ssim={1-ld.get('coarse_ssim',0):.4f}")

        scheduler.step()
        avg_loss = epoch_loss / max(batch_count, 1)
        elapsed = time.time() - t0
        writer.add_scalar('train/epoch_loss', avg_loss, epoch)
        writer.add_scalar('train/lr', scheduler.get_last_lr()[0], epoch)
        print(f"Epoch {epoch}: avg_loss={avg_loss:.4f} time={elapsed:.1f}s lr={scheduler.get_last_lr()[0]:.2e}")

        # Validation
        if epoch % args.val_every == 0 or epoch == args.epochs - 1:
            val_loss, val_ssim = validate(model, val_loader, criterion, device)
            writer.add_scalar('val/loss', val_loss, epoch)
            writer.add_scalar('val/ssim', val_ssim, epoch)
            print(f"  VAL loss={val_loss:.4f} ssim={val_ssim:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch, 'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                }, os.path.join(args.save_dir, 'flow_model_best.pth'))
                print(f"  Saved best model (val_loss={best_val_loss:.4f})")

        # Periodic save
        if epoch % args.save_every == 0:
            torch.save({
                'epoch': epoch, 'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
            }, os.path.join(args.save_dir, f'flow_model_epoch{epoch}.pth'))

    writer.close()
    print(f"\nStage 1 complete. Best val loss: {best_val_loss:.4f}")
    print(f"Next: python train_stage2.py --resume checkpoints/flow_model_best.pth")

if __name__ == '__main__':
    main()
