"""
Stage 3: End-to-end fine-tuning of both models with very low learning rate.
Both flow + refinement models are unfrozen and trained jointly.
"""
import os, argparse, time
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from models.pipeline import SatelliteInterpolator
from models.losses import TwoModelLoss
from dataset import SatelliteTripletDataset, SyntheticSatelliteDataset

def parse_args():
    p = argparse.ArgumentParser(description='Stage 3: End-to-End Fine-tuning')
    p.add_argument('--data_dir', type=str, default='data/processed')
    p.add_argument('--synthetic', action='store_true')
    p.add_argument('--resume', type=str, required=True, help='Stage 2 checkpoint')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--crop_size', type=int, default=256)
    p.add_argument('--num_workers', type=int, default=2)
    p.add_argument('--save_dir', type=str, default='checkpoints')
    p.add_argument('--log_dir', type=str, default='runs/stage3_e2e')
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

    # Load Stage 2 weights
    model = SatelliteInterpolator().to(device)
    ckpt = torch.load(args.resume, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    print(f"Loaded Stage 2 weights from {args.resume}")

    # UNFREEZE everything
    model.unfreeze_flow_model()
    model.summary()

    # Stage 3 loss — all components
    criterion = TwoModelLoss(stage=2, lambda_coarse=0.2, lambda_refined=1.0).to(device)

    # Differential LR: flow model gets lower LR
    optimizer = torch.optim.AdamW([
        {'params': model.flow_model.parameters(), 'lr': args.lr * 0.1},
        {'params': model.refinement_model.parameters(), 'lr': args.lr},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=1e-7)
    writer = SummaryWriter(args.log_dir)
    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        model.train()
        epoch_loss, bc = 0, 0
        t0 = time.time()

        for i, (f0, gt, f1) in enumerate(train_loader):
            f0, gt, f1 = f0.to(device), gt.to(device), f1.to(device)
            out = model(f0, f1, t=0.5, refine=True)
            loss, ld = criterion(out, gt)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            epoch_loss += loss.item(); bc += 1

            if i % 20 == 0:
                step = epoch * len(train_loader) + i
                for k, v in ld.items():
                    writer.add_scalar(f'train/{k}', v, step)

        scheduler.step()
        print(f"Epoch {epoch}: loss={epoch_loss/max(bc,1):.4f} time={time.time()-t0:.1f}s")

        if epoch % 5 == 0 or epoch == args.epochs - 1:
            model.eval()
            vl, vs, n = 0, 0, 0
            with torch.no_grad():
                for f0, gt, f1 in val_loader:
                    f0, gt, f1 = f0.to(device), gt.to(device), f1.to(device)
                    out = model(f0, f1, refine=True)
                    loss, ld = criterion(out, gt)
                    vl += loss.item()*f0.size(0)
                    vs += (1-ld.get('refined_ssim',0))*f0.size(0)
                    n += f0.size(0)
            vl, vs = vl/n, vs/n
            writer.add_scalar('val/loss', vl, epoch)
            writer.add_scalar('val/ssim', vs, epoch)
            print(f"  VAL loss={vl:.4f} ssim={vs:.4f}")
            if vl < best_val_loss:
                best_val_loss = vl
                torch.save({'epoch': epoch, 'model_state': model.state_dict(),
                            'best_val_loss': best_val_loss},
                           os.path.join(args.save_dir, 'e2e_model_best.pth'))
                print(f"  Saved best E2E model")

    writer.close()
    print(f"\nStage 3 (E2E) complete. Best val loss: {best_val_loss:.4f}")
    print(f"Final model: checkpoints/e2e_model_best.pth")

if __name__ == '__main__':
    main()
