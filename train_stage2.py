"""
Stage 2: Train the Refinement Model (Model 2) with frozen Flow Model.
Uses full loss: L1 + SSIM + Perceptual + Edge + Residual Regularization.
"""
import os, argparse, time
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from models.pipeline import SatelliteInterpolator
from models.losses import TwoModelLoss
from dataset import SatelliteTripletDataset, SyntheticSatelliteDataset

def parse_args():
    p = argparse.ArgumentParser(description='Stage 2: Train Refinement Model')
    p.add_argument('--data_dir', type=str, default='data/processed')
    p.add_argument('--synthetic', action='store_true')
    p.add_argument('--resume', type=str, required=True, help='Stage 1 checkpoint')
    p.add_argument('--epochs', type=int, default=80)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--crop_size', type=int, default=256)
    p.add_argument('--num_workers', type=int, default=2)
    p.add_argument('--save_dir', type=str, default='checkpoints')
    p.add_argument('--log_dir', type=str, default='runs/stage2')
    p.add_argument('--accum_steps', type=int, default=2, help='Gradient accumulation steps')
    p.add_argument('--use_amp', action='store_true', help='Use automatic mixed precision')
    return p.parse_args()

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss, total_ssim, n = 0, 0, 0
    with torch.no_grad():
        for f0, gt, f1 in val_loader:
            f0, gt, f1 = f0.to(device), gt.to(device), f1.to(device)
            out = model(f0, f1, t=0.5, refine=True)
            loss, ld = criterion(out, gt)
            total_loss += loss.item() * f0.size(0)
            total_ssim += (1 - ld.get('refined_ssim', 0)) * f0.size(0)
            n += f0.size(0)
    return total_loss / n, total_ssim / n

def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Dataset
    if args.synthetic:
        train_ds = SyntheticSatelliteDataset(2000, args.crop_size)
        val_ds = SyntheticSatelliteDataset(200, args.crop_size)
    else:
        train_ds = SatelliteTripletDataset(args.data_dir, 'train', args.crop_size)
        val_ds = SatelliteTripletDataset(args.data_dir, 'val', crop_size=None, augment=False)

    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    # Model — load Stage 1 weights
    model = SatelliteInterpolator().to(device)
    ckpt = torch.load(args.resume, map_location=device)
    model.load_state_dict(ckpt['model_state'], strict=False)
    print(f"Loaded Stage 1 weights from {args.resume}")

    # FREEZE flow model
    model.freeze_flow_model()
    model.summary()

    # Loss (Stage 2 — all losses active)
    criterion = TwoModelLoss(stage=2, lambda_coarse=0.3).to(device)

    # Only optimize refinement model parameters
    optimizer = torch.optim.AdamW(model.refinement_model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=1e-6)
    writer = SummaryWriter(args.log_dir)
    best_val_loss = float('inf')

    # AMP Setup
    use_amp = args.use_amp and device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    accum_steps = max(1, args.accum_steps)

    print(f"AMP: {'ENABLED' if use_amp else 'DISABLED'} | Gradient Accumulation Steps: {accum_steps}")

    for epoch in range(args.epochs):
        model.train()
        model.flow_model.eval()  # Keep flow model in eval mode
        epoch_loss, batch_count = 0, 0
        t0 = time.time()

        optimizer.zero_grad()

        for i, (f0, gt, f1) in enumerate(train_loader):
            f0, gt, f1 = f0.to(device), gt.to(device), f1.to(device)
            
            with torch.cuda.amp.autocast(enabled=use_amp):
                out = model(f0, f1, t=0.5, refine=True)
                loss, ld = criterion(out, gt)
                # Normalize loss for accumulation
                loss = loss / accum_steps

            scaler.scale(loss).backward()
            epoch_loss += loss.item() * accum_steps
            batch_count += 1

            # Optimize step after accumulating enough gradients
            if (i + 1) % accum_steps == 0 or (i + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.refinement_model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            if i % 20 == 0:
                step = epoch * len(train_loader) + i
                for k, v in ld.items():
                    writer.add_scalar(f'train/{k}', v, step)
                print(f"  [{epoch}/{args.epochs}] batch {i}/{len(train_loader)} "
                      f"loss={loss.item() * accum_steps:.4f} "
                      f"refined_ssim={1-ld.get('refined_ssim',0):.4f} "
                      f"residual_reg={ld.get('residual_reg',0):.6f}")

        scheduler.step()
        avg_loss = epoch_loss / max(batch_count, 1)
        print(f"Epoch {epoch}: avg_loss={avg_loss:.4f} time={time.time()-t0:.1f}s")

        # Validation every 5 epochs
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            val_loss, val_ssim = validate(model, val_loader, criterion, device)
            writer.add_scalar('val/loss', val_loss, epoch)
            writer.add_scalar('val/ssim', val_ssim, epoch)
            print(f"  VAL loss={val_loss:.4f} ssim={val_ssim:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch, 'model_state': model.state_dict(),
                    'best_val_loss': best_val_loss,
                }, os.path.join(args.save_dir, 'refinement_model_best.pth'))
                print(f"  Saved best model")

    writer.close()
    print(f"\nStage 2 complete. Best val loss: {best_val_loss:.4f}")
    print(f"Next: python train_e2e.py --resume checkpoints/refinement_model_best.pth")

if __name__ == '__main__':
    main()
