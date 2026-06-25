"""
Inference script for the 2-model satellite frame interpolation pipeline.
Supports single-pair and recursive interpolation, outputs .npy and .png.
"""
import os, argparse, glob, time
import numpy as np
import torch
from PIL import Image
from models.pipeline import SatelliteInterpolator

def parse_args():
    p = argparse.ArgumentParser(description='Satellite Frame Interpolation — Inference')
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--input_dir', type=str, required=True, help='Dir with sorted .npy frames')
    p.add_argument('--output_dir', type=str, default='output/interpolated')
    p.add_argument('--depth', type=int, default=1, help='Recursive depth (1=1x, 2=3x, 3=7x)')
    p.add_argument('--image_size', type=int, default=512)
    p.add_argument('--save_png', action='store_true', help='Also save as PNG images')
    p.add_argument('--compare', action='store_true', help='Save side-by-side comparison')
    return p.parse_args()

def load_frame(path, size=512):
    frame = np.load(path).astype(np.float32)
    if frame.ndim == 3:
        frame = frame[:, :, 0]
    if frame.shape[0] != size or frame.shape[1] != size:
        img = Image.fromarray((frame * 255).astype(np.uint8))
        img = img.resize((size, size), Image.BICUBIC)
        frame = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(frame).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

def save_frame(tensor, path, save_png=False):
    frame = tensor.squeeze().cpu().numpy()
    np.save(path, frame)
    if save_png:
        img = Image.fromarray((frame * 255).astype(np.uint8))
        img.save(path.replace('.npy', '.png'))

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = SatelliteInterpolator().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f"Loaded model from {args.checkpoint}")
    model.summary()

    # Load input frames
    frame_paths = sorted(glob.glob(os.path.join(args.input_dir, '*.npy')))
    print(f"Found {len(frame_paths)} input frames")

    if len(frame_paths) < 2:
        print("Need at least 2 frames for interpolation!")
        return

    # Process consecutive pairs
    all_output_frames = []
    total_time = 0

    for i in range(len(frame_paths) - 1):
        f0 = load_frame(frame_paths[i], args.image_size).to(device)
        f1 = load_frame(frame_paths[i + 1], args.image_size).to(device)

        t0 = time.time()
        with torch.no_grad():
            if args.depth == 1:
                mid = model.inference(f0, f1, t=0.5)
                interpolated = [mid]
            else:
                interpolated = model.recursive_interpolate(f0, f1, depth=args.depth)
        elapsed = time.time() - t0
        total_time += elapsed

        # Save original frame
        save_frame(f0, os.path.join(args.output_dir, f'{i*100:06d}_original.npy'), args.save_png)
        all_output_frames.append(f0)

        # Save interpolated frames
        for j, frame in enumerate(interpolated):
            idx = i * 100 + (j + 1) * (100 // (len(interpolated) + 1))
            save_frame(frame, os.path.join(args.output_dir, f'{idx:06d}_interp.npy'), args.save_png)
            all_output_frames.append(frame)

        print(f"Pair {i}-{i+1}: {len(interpolated)} frames interpolated in {elapsed:.3f}s")

    # Save last original frame
    f_last = load_frame(frame_paths[-1], args.image_size).to(device)
    save_frame(f_last, os.path.join(args.output_dir, f'{(len(frame_paths)-1)*100:06d}_original.npy'), args.save_png)

    print(f"\nDone! {len(all_output_frames)} total output frames")
    print(f"Total inference time: {total_time:.2f}s")
    print(f"Saved to: {args.output_dir}")

if __name__ == '__main__':
    main()
