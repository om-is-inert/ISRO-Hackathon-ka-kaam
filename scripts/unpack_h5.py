import h5py
import numpy as np
import os
from tqdm import tqdm

# Paths
h5_path = "hurricane_dataset.h5"
output_dir = "data/processed/frames"

# Create the directories if they don't exist
os.makedirs(output_dir, exist_ok=True)

print("Opening the .h5 vault...")

# Read the data
with h5py.File(h5_path, 'r') as h5f:
    images = h5f['images'][:]
    
print(f"Found {len(images)} matrices. Unpacking to .npy format...")

# Loop through and save each matrix as a NumPy array
for i in tqdm(range(len(images)), desc="Extracting"):
    frame_data = images[i]
    # Format name with leading zeros (e.g., frame_0001.npy) so they sort correctly
    save_path = os.path.join(output_dir, f"frame_{i:04d}.npy")
    np.save(save_path, frame_data)

print(f"\nSuccess! All frames extracted to {output_dir}")
print("You are ready to run the triplet creation script.")