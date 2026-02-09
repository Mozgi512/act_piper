#!/usr/bin/env python3
"""
Filter out failed episodes (timeout episodes) from a dataset.
Failed episodes are identified by having length >= max_length threshold.
"""

import argparse
import h5py
import numpy as np
import os
import shutil
from tqdm import tqdm
import glob


def filter_dataset(input_dir, output_dir, max_length):
    """
    Filter out episodes that are too long (likely failed/timeout episodes).
    
    Args:
        input_dir: Directory containing input HDF5 files
        output_dir: Directory to save filtered HDF5 files
        max_length: Maximum episode length threshold (episodes >= this are excluded)
    """
    # Get all HDF5 files
    input_files = sorted(glob.glob(os.path.join(input_dir, '*.hdf5')))
    
    if not input_files:
        print(f"No HDF5 files found in {input_dir}")
        return
    
    print(f"Found {len(input_files)} HDF5 files in {input_dir}")
    print(f"Filtering episodes with length >= {max_length}")
    print()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter episodes
    kept_count = 0
    filtered_count = 0
    
    for input_file in tqdm(input_files, desc="Filtering episodes"):
        with h5py.File(input_file, 'r') as hf:
            episode_length = hf['/action'].shape[0]
            
            if episode_length < max_length:
                # Keep this episode
                output_file = os.path.join(output_dir, f'episode_{kept_count}.hdf5')
                shutil.copy2(input_file, output_file)
                kept_count += 1
            else:
                # Filter out this episode
                filtered_count += 1
    
    print()
    print(f"Filtering complete!")
    print(f"  Kept: {kept_count} episodes")
    print(f"  Filtered out: {filtered_count} episodes (length >= {max_length})")
    print(f"  Output directory: {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Filter out failed episodes from dataset')
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory containing HDF5 files')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for filtered files')
    parser.add_argument('--max_length', type=int, default=2900, help='Maximum episode length (default: 2900)')
    
    args = parser.parse_args()
    
    filter_dataset(args.input_dir, args.output_dir, args.max_length)
