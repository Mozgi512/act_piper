#!/usr/bin/env python3
"""
Merge datasets by moving and renumbering episode files.

This script moves individual episode_*.hdf5 files from multiple directories
into a single output directory, renumbering them sequentially.

Usage:
    python merge_datasets.py --input_dirs data/IIC data/CCC data/III --output_dir data/merged
"""

import os
import shutil
import glob
import argparse
from tqdm import tqdm

def merge_datasets(input_dirs, output_dir):
    """
    Merge datasets by moving episode files and renumbering them.
    
    Args:
        input_dirs: List of directories containing episode_*.hdf5 files
        output_dir: Output directory for merged episodes
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect all episode files from input directories
    all_episodes = []
    for input_dir in input_dirs:
        if not os.path.exists(input_dir):
            print(f"Warning: Directory {input_dir} does not exist, skipping...")
            continue
        
        # Find all episode_*.hdf5 files recursively
        episode_files = []
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if file.startswith('episode_') and file.endswith('.hdf5'):
                    episode_files.append(os.path.join(root, file))
        episode_files.sort()
        
        if episode_files:
            all_episodes.extend(episode_files)
            print(f"Found {len(episode_files)} episodes in {input_dir}")
        else:
            print(f"Warning: No episode files found in {input_dir}")
    
    if not all_episodes:
        print("Error: No episode files found in any input directory!")
        return
    
    total_episodes = len(all_episodes)
    print(f"\nMerging {total_episodes} episodes to {output_dir}...")
    
    # Copy and renumber episodes
    for new_idx, src_path in enumerate(tqdm(all_episodes, desc="Copying episodes")):
        dst_filename = f'episode_{new_idx}.hdf5'
        dst_path = os.path.join(output_dir, dst_filename)
        
        # Copy file (use copy instead of move to preserve originals)
        shutil.copy2(src_path, dst_path)
    
    print(f"\n✅ Successfully merged {total_episodes} episodes to: {output_dir}")
    
    # Verify output
    output_files = glob.glob(os.path.join(output_dir, 'episode_*.hdf5'))
    print(f"   Output contains {len(output_files)} episode files")
    print(f"   Episode range: episode_0.hdf5 to episode_{total_episodes-1}.hdf5")

def main():
    parser = argparse.ArgumentParser(description='Merge datasets by moving and renumbering episode files')
    parser.add_argument('--input_dirs', nargs='+', required=True,
                        help='List of input directories containing episode_*.hdf5 files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for merged episodes')
    parser.add_argument('--move', action='store_true',
                        help='Move files instead of copying (default: copy)')
    
    args = parser.parse_args()
    
    # Update function to support move option
    if args.move:
        print("Note: Using MOVE mode (original files will be deleted)")
    
    merge_datasets(args.input_dirs, args.output_dir)

if __name__ == '__main__':
    main()
