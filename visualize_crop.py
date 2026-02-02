import h5py
import numpy as np
import cv2
import os
import argparse
import sys
from utils import apply_rgb_mask_to_strip

def visualize(dataset_dir):
    files = sorted([f for f in os.listdir(dataset_dir) if f.endswith('.hdf5')])
    if not files:
        print("No HDF5 files found")
        return

    fpath = os.path.join(dataset_dir, files[0])
    print(f"Reading {fpath}")
    
    with h5py.File(fpath, 'r') as root:
        # Get first image from 'top' camera
        img = root['/observations/images/top'][0] # H W C
        
        # Save Original
        cv2.imwrite('crop_debug_original.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        
        # Apply Right Arm Logic (from independent_imitate_episodes.py)
        # H W C
        h, w, c = img.shape
        print(f"Original Shape: {h}x{w}x{c}")
        
        # Right Arm Logic
        offset = int(40 * (w / 640))
        start = w//2 - offset
        end = w - offset
        print(f"Crop Range: {start} to {end} (Offset: {offset})")
        
        curr_image = img[:, start:end, :]
        print(f"Cropped Shape: {curr_image.shape}")
        
        # Mask Logic
        # apply_rgb_mask_to_strip expects HWC (which is what we have)
        masked_image = apply_rgb_mask_to_strip(curr_image.copy(), strip_width=offset)
        
        cv2.imwrite('crop_debug_right.png', cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR))
        print("Saved crop_debug_original.png and crop_debug_right.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir')
    args = parser.parse_args()
    visualize(args.dataset_dir)
