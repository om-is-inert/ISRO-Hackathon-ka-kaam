"""
PyTorch Dataset for satellite frame interpolation training.
Supports real (preprocessed .npy triplets) and synthetic (moving blobs) data.

Optimized augmentations for satellite TIR imagery:
  - Flip (horizontal/vertical) + 90° rotation
  - Temporal reversal (swap f0 ↔ f1)
  - Brightness jitter (simulates different thermal conditions)
  - Gaussian noise injection (simulates sensor noise)
  - Multi-scale blobs in synthetic data (more realistic)
"""
import os, json, random
import numpy as np
import torch
from torch.utils.data import Dataset

class SatelliteTripletDataset(Dataset):
    """Loads (frame0, gt, frame1) .npy triplets from disk."""
    def __init__(self, data_dir, split='train', crop_size=256, augment=True):
        self.data_dir = os.path.join(data_dir, split)
        self.crop_size = crop_size if split == 'train' else None
        self.augment = augment and split == 'train'
        triplets_file = os.path.join(self.data_dir, 'triplets.json')
        if os.path.exists(triplets_file):
            with open(triplets_file, 'r') as f:
                self.triplets = json.load(f)
        else:
            self.triplets = self._discover_triplets()

    def _discover_triplets(self):
        triplets = []
        if not os.path.exists(self.data_dir):
            return triplets
        for d in sorted(os.listdir(self.data_dir)):
            td = os.path.join(self.data_dir, d)
            if os.path.isdir(td):
                paths = [os.path.join(td, n) for n in ['frame0.npy','gt.npy','frame1.npy']]
                if all(os.path.exists(p) for p in paths):
                    triplets.append({'frame0':paths[0],'gt':paths[1],'frame1':paths[2]})
        return triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        t = self.triplets[idx]
        f0 = np.load(t['frame0']).astype(np.float32)
        gt = np.load(t['gt']).astype(np.float32)
        f1 = np.load(t['frame1']).astype(np.float32)
        if f0.ndim == 3: f0, gt, f1 = f0[:,:,0], gt[:,:,0], f1[:,:,0]
        if self.crop_size:
            h,w = f0.shape
            if h > self.crop_size and w > self.crop_size:
                y,x = random.randint(0,h-self.crop_size), random.randint(0,w-self.crop_size)
                s = (slice(y,y+self.crop_size), slice(x,x+self.crop_size))
                f0, gt, f1 = f0[s], gt[s], f1[s]
        if self.augment:
            # Spatial augmentations
            if random.random()>0.5: f0,gt,f1 = np.flip(f0,1).copy(),np.flip(gt,1).copy(),np.flip(f1,1).copy()
            if random.random()>0.5: f0,gt,f1 = np.flip(f0,0).copy(),np.flip(gt,0).copy(),np.flip(f1,0).copy()
            k = random.randint(0,3)
            if k: f0,gt,f1 = np.rot90(f0,k).copy(),np.rot90(gt,k).copy(),np.rot90(f1,k).copy()
            # Temporal reversal (swap frames — the model should be symmetrical)
            if random.random()>0.5: f0,f1 = f1,f0
            # Brightness jitter (simulates different thermal conditions)
            if random.random() > 0.5:
                factor = random.uniform(0.85, 1.15)
                f0, gt, f1 = f0 * factor, gt * factor, f1 * factor
                f0 = np.clip(f0, 0, 1)
                gt = np.clip(gt, 0, 1)
                f1 = np.clip(f1, 0, 1)
            # Gaussian noise injection (simulates sensor noise)
            if random.random() > 0.5:
                sigma = random.uniform(0.005, 0.02)
                noise = np.random.normal(0, sigma, f0.shape).astype(np.float32)
                f0, gt, f1 = f0 + noise, gt + noise, f1 + noise
                f0 = np.clip(f0, 0, 1)
                gt = np.clip(gt, 0, 1)
                f1 = np.clip(f1, 0, 1)
        return (torch.from_numpy(f0).unsqueeze(0), torch.from_numpy(gt).unsqueeze(0),
                torch.from_numpy(f1).unsqueeze(0))

class SyntheticSatelliteDataset(Dataset):
    """
    Improved synthetic dataset — multi-scale Gaussian blobs with
    background texture for more realistic training signal.
    """
    def __init__(self, num_samples=1000, image_size=256, num_blobs=5):
        self.n, self.sz, self.nb = num_samples, image_size, num_blobs
    def __len__(self):
        return self.n
    def __getitem__(self, idx):
        H = W = self.sz
        # Create a subtle background texture (not pure black)
        bg = np.random.uniform(0.02, 0.08, (H, W)).astype(np.float32)
        f0 = bg.copy()
        gt = bg.copy()
        f1 = bg.copy()
        yc, xc = np.mgrid[0:H, 0:W].astype(np.float32)
        for _ in range(self.nb):
            cx, cy = random.uniform(50, W-50), random.uniform(50, H-50)
            # Multi-scale: mix of small and large blobs
            r = random.uniform(15, 70)
            intensity = random.uniform(0.3, 1.0)
            vx, vy = random.uniform(-15, 15), random.uniform(-15, 15)
            f0 += intensity * np.exp(-((xc-cx)**2 + (yc-cy)**2) / (2*r**2))
            gt += intensity * np.exp(-((xc-cx-vx/2)**2 + (yc-cy-vy/2)**2) / (2*r**2))
            f1 += intensity * np.exp(-((xc-cx-vx)**2 + (yc-cy-vy)**2) / (2*r**2))
        # Add sensor noise
        noise = lambda: np.random.normal(0, 0.02, (H, W)).astype(np.float32)
        f0, gt, f1 = [np.clip(x + noise(), 0, 1) for x in [f0, gt, f1]]
        return tuple(torch.from_numpy(x).unsqueeze(0) for x in [f0, gt, f1])
