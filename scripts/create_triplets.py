"""
Create training triplets from preprocessed frames.
For GOES-19 (10-min cadence): stride=2 creates (t0, t10, t20) triplets
  where t10 is the ground truth for interpolation between t0 and t20.

Usage:
    python scripts/create_triplets.py --frames data/processed/frames/ --output data/processed/
"""
import os, json, argparse, glob, random
import numpy as np
from shutil import copy2

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--frames', type=str, required=True, help='Dir with .npy frames')
    p.add_argument('--output', type=str, default='data/processed')
    p.add_argument('--stride', type=int, default=2, help='Frame stride (2=skip 1 for GT)')
    p.add_argument('--train_ratio', type=float, default=0.8)
    p.add_argument('--val_ratio', type=float, default=0.1)
    args = p.parse_args()

    frames = sorted(glob.glob(os.path.join(args.frames, '*.npy')))
    print(f"Found {len(frames)} frames")

    # Create triplets
    triplets = []
    stride = args.stride
    for i in range(0, len(frames) - stride, 1):
        triplets.append({
            'frame0': frames[i],
            'gt': frames[i + stride // 2],
            'frame1': frames[i + stride],
        })
    print(f"Created {len(triplets)} triplets (stride={stride})")

    # Split
    random.seed(42)
    random.shuffle(triplets)
    n = len(triplets)
    n_train = int(n * args.train_ratio)
    n_val = int(n * args.val_ratio)
    splits = {
        'train': triplets[:n_train],
        'val': triplets[n_train:n_train + n_val],
        'test': triplets[n_train + n_val:],
    }

    # Save triplets as directories with symlinks
    for split_name, split_triplets in splits.items():
        split_dir = os.path.join(args.output, split_name)
        os.makedirs(split_dir, exist_ok=True)

        for idx, t in enumerate(split_triplets):
            triplet_dir = os.path.join(split_dir, f'{idx:06d}')
            os.makedirs(triplet_dir, exist_ok=True)
            # Copy frames to triplet directory
            for key in ['frame0', 'gt', 'frame1']:
                src = t[key]
                dst = os.path.join(triplet_dir, f'{key}.npy')
                if not os.path.exists(dst):
                    copy2(src, dst)

        # Also save triplets.json for reference
        json_triplets = [{'frame0': os.path.join(split_dir, f'{i:06d}', 'frame0.npy'),
                          'gt': os.path.join(split_dir, f'{i:06d}', 'gt.npy'),
                          'frame1': os.path.join(split_dir, f'{i:06d}', 'frame1.npy')}
                         for i in range(len(split_triplets))]
        with open(os.path.join(split_dir, 'triplets.json'), 'w') as f:
            json.dump(json_triplets, f)

        print(f"  {split_name}: {len(split_triplets)} triplets → {split_dir}")

if __name__ == '__main__':
    main()
