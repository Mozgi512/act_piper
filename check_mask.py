import h5py
import numpy as np
import os
import argparse

def check_mask(dataset_dir):
    files = sorted([f for f in os.listdir(dataset_dir) if f.endswith('.hdf5')])
    if not files:
        print("No HDF5 files found")
        return

    fpath = os.path.join(dataset_dir, files[0])
    print(f"Reading {fpath}")
    
    with h5py.File(fpath, 'r') as root:
        img = root['/observations/images/top'][0] # H W C
        h, w, c = img.shape
        print(f"Shape: {h}x{w}x{c}")
        
        # Check Left 40px
        left_strip = img[:, :40, :]
        mean_val = np.mean(left_strip)
        std_val = np.std(left_strip)
        print(f"Left 40px Mean: {mean_val}")
        print(f"Left 40px Std: {std_val}")
        
        if mean_val < 5: # Threshold for black
            print("Left 40px appears MASKED (Black)")
        else:
            print("Left 40px appears UNMASKED (Content visible)")
            
        # Check Middle
        mid_strip = img[:, 60:100, :]
        print(f"Middle Strip Mean: {np.mean(mid_strip)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir')
    args = parser.parse_args()
    check_mask(args.dataset_dir)
