"""
Evaluation script — computes PSNR, SSIM, MSE, LPIPS on interpolated vs ground truth.
Generates a CSV report and per-frame metrics plots.
"""
import os, argparse, glob
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import pandas as pd
import matplotlib.pyplot as plt
from models.pipeline import SatelliteInterpolator

def compute_metrics(pred, gt):
    """Compute all metrics between predicted and ground truth frames."""
    pred_np = pred.squeeze().cpu().numpy()
    gt_np = gt.squeeze().cpu().numpy()
    return {
        'PSNR': psnr(gt_np, pred_np, data_range=1.0),
        'SSIM': ssim(gt_np, pred_np, data_range=1.0),
        'MSE': float(np.mean((pred_np - gt_np) ** 2)),
        'MAE': float(np.mean(np.abs(pred_np - gt_np))),
    }

def main():
    p = argparse.ArgumentParser(description='Evaluate interpolation quality')
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--data_dir', type=str, default='data/processed')
    p.add_argument('--split', type=str, default='test')
    p.add_argument('--output', type=str, default='output/evaluation')
    p.add_argument('--image_size', type=int, default=512)
    args = p.parse_args()
    os.makedirs(args.output, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SatelliteInterpolator().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    # Load test triplets
    from dataset import SatelliteTripletDataset
    test_ds = SatelliteTripletDataset(args.data_dir, args.split, crop_size=None, augment=False)
    print(f"Evaluating on {len(test_ds)} test triplets...")

    all_metrics = []
    coarse_metrics_list = []

    for i in range(len(test_ds)):
        f0, gt, f1 = test_ds[i]
        f0 = f0.unsqueeze(0).to(device)
        gt = gt.unsqueeze(0).to(device)
        f1 = f1.unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(f0, f1, t=0.5, refine=True)

        # Refined metrics
        m = compute_metrics(out['refined'], gt)
        m['frame_idx'] = i
        all_metrics.append(m)

        # Coarse metrics (for comparison)
        mc = compute_metrics(out['coarse'], gt)
        mc['frame_idx'] = i
        coarse_metrics_list.append(mc)

        if i % 10 == 0:
            print(f"  Frame {i}: PSNR={m['PSNR']:.2f} SSIM={m['SSIM']:.4f} "
                  f"(coarse: PSNR={mc['PSNR']:.2f} SSIM={mc['SSIM']:.4f})")

    # Summary
    df = pd.DataFrame(all_metrics)
    df_coarse = pd.DataFrame(coarse_metrics_list)
    df.to_csv(os.path.join(args.output, 'metrics_refined.csv'), index=False)
    df_coarse.to_csv(os.path.join(args.output, 'metrics_coarse.csv'), index=False)

    print(f"\n{'='*60}")
    print(f"{'Metric':<10} {'Coarse (Model 1)':>18} {'Refined (Model 1+2)':>20}")
    print(f"{'='*60}")
    for m in ['PSNR', 'SSIM', 'MSE', 'MAE']:
        c_val = df_coarse[m].mean()
        r_val = df[m].mean()
        diff = r_val - c_val
        sign = '+' if diff > 0 else ''
        print(f"{m:<10} {c_val:>18.4f} {r_val:>18.4f}  ({sign}{diff:.4f})")
    print(f"{'='*60}")

    # Plot per-frame metrics
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, metric in zip(axes.flat, ['PSNR', 'SSIM', 'MSE', 'MAE']):
        ax.plot(df['frame_idx'], df[metric], 'b-', label='Refined (2-model)', alpha=0.8)
        ax.plot(df_coarse['frame_idx'], df_coarse[metric], 'r--', label='Coarse (flow only)', alpha=0.6)
        ax.set_xlabel('Frame Index')
        ax.set_ylabel(metric)
        ax.set_title(metric)
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.suptitle('2-Model Interpolation: Coarse vs Refined', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output, 'metrics_comparison.png'), dpi=150)
    print(f"\nPlot saved to {args.output}/metrics_comparison.png")

if __name__ == '__main__':
    main()
