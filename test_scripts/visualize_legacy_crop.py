import h5py
import numpy as np
import cv2
import os
import argparse
from utils import apply_rgb_mask_to_strip

def visualize_legacy(dataset_dir):
    files = sorted([f for f in os.listdir(dataset_dir) if f.endswith('.hdf5')])
    if not files:
        print("No HDF5 files found")
        return

    fpath = os.path.join(dataset_dir, files[0])
    print(f"Reading {fpath}")
    
    with h5py.File(fpath, 'r') as root:
        img = root['/observations/images/top'][0] # H W C
        # Force 240x160 check (or whatever it is)
        h, w, c = img.shape
        print(f"Input Shape: {h}x{w}x{c}")
        
        cv2.imwrite('legacy_input.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        
        # --- LEGACY LOGIC START ---
        # Calculate offset based on width (base 640 -> 40)
        # Note: In the bug, 'w' was the width of the loaded image (160)
        offset = int(40 * (w / 640))
        
        # Shift based on width (adaptive)
        start = w//2 - offset
        end = w - offset
        
        print(f"Legacy Calculation:")
        print(f"  Width (w): {w}")
        print(f"  Offset: {offset} (int(40 * {w}/640))")
        print(f"  Start: {start} ({w}//2 - {offset})")
        print(f"  End: {end}   ({w} - {offset})")
        
        # Right Arm: masks the LEFT side (overlap region)
        curr_image = img[:, start:end, :].copy()
        print(f"  Cropped Shape: {curr_image.shape}")
        
        # Apply RGB mask
        curr_image = apply_rgb_mask_to_strip(curr_image, strip_width=offset)
        # --- LEGACY LOGIC END ---
        
        cv2.imwrite('legacy_output_masked.png', cv2.cvtColor(curr_image, cv2.COLOR_RGB2BGR))
        print("Saved legacy_output_masked.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir')
    args = parser.parse_args()
    visualize_legacy(args.dataset_dir)
