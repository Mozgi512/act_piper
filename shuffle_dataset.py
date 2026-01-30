#!/usr/bin/env python3
"""
Shuffle episode files in a dataset directory.

This script renames episode_*.hdf5 files in a random order.
To safely rename without overwriting, it uses a temporary directory or temporary names.

Usage:
    python shuffle_dataset.py --dataset_dir data/my_dataset
"""

import os
import glob
import argparse
import random
import shutil
from tqdm import tqdm

def shuffle_dataset(dataset_dir, seed=None):
    if seed is not None:
        random.seed(seed)
        
    print(f"Shuffling dataset in: {dataset_dir}")
    
    # 1. Find all episode files
    files = glob.glob(os.path.join(dataset_dir, 'episode_*.hdf5'))
    if not files:
        print("No episode files found.")
        return

    num_files = len(files)
    print(f"Found {num_files} episodes.")
    
    # 2. Generate random mapping
    indices = list(range(num_files))
    random.shuffle(indices)
    
    # 3. Rename to temporary names to avoid conflicts
    # e.g. episode_0.hdf5 -> temp_123.hdf5
    # We map old_path -> new_idx
    
    # Create a temp subdirectory to be safe? Or just rename in place with prefix.
    # In-place rename with prefix is safer against interruption than moving to another volume.
    
    print("Renaming to temporary names...")
    temp_files = []
    for src in tqdm(files, desc="Temp rename"):
        dirname, basename = os.path.split(src)
        temp_name = f"temp_{os.path.basename(src)}"
        dst = os.path.join(dirname, temp_name)
        os.rename(src, dst)
        temp_files.append(dst)
        
    # 4. Rename from temp to new random indices
    print("Renaming to new shuffled indices...")
    # temp_files list corresponds to the original files (in whatever order glob returned)
    # indices list is scrambled 0..N-1
    
    for i, temp_path in enumerate(tqdm(temp_files, desc="Final rename")):
        new_idx = indices[i]
        dirname = os.path.dirname(temp_path)
        new_name = f"episode_{new_idx}.hdf5"
        dst = os.path.join(dirname, new_name)
        os.rename(temp_path, dst)
        
    print("✅ Shuffle completed.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, required=True, help='Directory containing episode_*.hdf5 files')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.dataset_dir):
        print(f"Error: Directory {args.dataset_dir} does not exist.")
        exit(1)
        
    shuffle_dataset(args.dataset_dir, args.seed)
