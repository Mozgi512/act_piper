import h5py
import numpy as np
import os
import argparse

def verify_masking(dataset_dir, episode_idx=0):
    filepath = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return

    print(f"Verifying {filepath}...")
    with h5py.File(filepath, 'r') as f:
        # Check Left Phase 2 (Right Edge Masked)
        if 'phase2_left' in dataset_dir:
            print("Checking Left Phase 2 (Right Edge Masking)...")
            images = f['/observations/images']
            for cam_name in images:
                img_data = images[cam_name][0] # First frame
                h, w, c = img_data.shape
                print(f"  Camera: {cam_name}, Shape: {img_data.shape}")
                
                # Right strip 40px
                strip = img_data[:, -40:, :]
                
                # Check how many pixels are not black
                non_black = np.any(strip > 0, axis=2)
                count = np.sum(non_black)
                total = strip.shape[0] * strip.shape[1]
                percent = (count / total) * 100
                print(f"    Right 40px strip non-black pixels: {count}/{total} ({percent:.2f}%)")
                
                # Verify that non-black pixels are likely RGB (heuristic)
                # This is harder without the util logic, but low percentage suggests masking worked (assuming background is mostly not black)
                
        # Check Right Phase 2 (Left Edge Masked, Shifted)
        elif 'phase2_right' in dataset_dir:
            print("Checking Right Phase 2 (Left Edge Masking)...")
            images = f['/observations/images']
            for cam_name in images:
                img_data = images[cam_name][0]
                h, w, c = img_data.shape
                print(f"  Camera: {cam_name}, Shape: {img_data.shape}")
                
                # Left strip 40px (since it was shifted, identifying the masked region depends on the crop)
                # In process_coop_data.py: 
                # Crop Right: [:, 320-40, 640-40] -> 280:600. Width 320.
                # If offset > 0, apply mask to strip_width=offset (Left side of the crop)
                
                strip = img_data[:, :40, :]
                
                non_black = np.any(strip > 0, axis=2)
                count = np.sum(non_black)
                total = strip.shape[0] * strip.shape[1]
                percent = (count / total) * 100
                print(f"    Left 40px strip non-black pixels: {count}/{total} ({percent:.2f}%)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True)
    args = parser.parse_args()
    verify_masking(args.dir)
