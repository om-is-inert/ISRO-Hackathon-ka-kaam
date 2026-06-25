"""
Preprocess satellite data into normalized .npy frames.
Supports GOES-19 (.nc) and INSAT-3DS (.h5) formats.

Usage:
    python scripts/preprocess.py --source goes19 --input data/goes19/ --output data/processed/frames/
    python scripts/preprocess.py --source insat3ds --input data/insat3ds/ --output data/processed/frames/
"""
import os, argparse, glob
import numpy as np
from PIL import Image

def preprocess_goes19(nc_path, output_size=512):
    """Extract brightness temperature from GOES-19 Ch13 .nc file."""
    import xarray as xr
    ds = xr.open_dataset(nc_path)
    rad = ds['Rad'].values
    fk1 = ds['planck_fk1'].values
    fk2 = ds['planck_fk2'].values
    bc1 = ds['planck_bc1'].values
    bc2 = ds['planck_bc2'].values
    Tb = (fk2 / np.log((fk1 / rad) + 1) - bc1) / bc2
    Tb = np.clip(Tb, 180, 330)
    Tb_norm = (Tb - 180.0) / (330.0 - 180.0)
    # Handle NaN
    Tb_norm = np.nan_to_num(Tb_norm, nan=0.0)
    img = Image.fromarray((Tb_norm * 255).astype(np.uint8))
    img = img.resize((output_size, output_size), Image.BICUBIC)
    return np.array(img).astype(np.float32) / 255.0

def preprocess_insat3ds(h5_path, output_size=512):
    """Extract TIR1 brightness temperature from INSAT-3DS .h5 file."""
    import h5py
    from skimage.transform import resize
    with h5py.File(h5_path, 'r') as f:
        indices = np.array(f['IMG_TIR1'][0])
        temp_lut = np.array(f['IMG_TIR1_TEMP'])
        Tb = temp_lut[indices.astype(int)]
        Tb_norm = Tb - Tb.min()
        mx = Tb_norm.max()
        if mx > 0:
            Tb_norm = Tb_norm / mx
        Tb_resized = resize(Tb_norm, (output_size, output_size), anti_aliasing=True)
    return Tb_resized.astype(np.float32)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', choices=['goes19', 'insat3ds'], required=True)
    p.add_argument('--input', type=str, required=True)
    p.add_argument('--output', type=str, default='data/processed/frames')
    p.add_argument('--size', type=int, default=512)
    args = p.parse_args()
    os.makedirs(args.output, exist_ok=True)

    if args.source == 'goes19':
        files = sorted(glob.glob(os.path.join(args.input, '*.nc')))
        preprocess_fn = lambda f: preprocess_goes19(f, args.size)
    else:
        files = sorted(glob.glob(os.path.join(args.input, '*.h5')))
        preprocess_fn = lambda f: preprocess_insat3ds(f, args.size)

    print(f"Processing {len(files)} {args.source} files...")
    for i, filepath in enumerate(files):
        try:
            frame = preprocess_fn(filepath)
            out_path = os.path.join(args.output, f'{i:06d}.npy')
            np.save(out_path, frame)
            if i % 10 == 0:
                print(f"  [{i}/{len(files)}] {os.path.basename(filepath)} → {out_path}")
        except Exception as e:
            print(f"  Error processing {filepath}: {e}")
    print(f"Done! {len(files)} frames saved to {args.output}")

if __name__ == '__main__':
    main()
