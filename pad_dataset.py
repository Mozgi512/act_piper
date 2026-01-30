#!/usr/bin/env python3
"""
Pad episodes in HDF5 dataset to uniform length for batch training.

Usage:
    # Single file
    python pad_dataset.py --input_file data/merged/dataset.hdf5 --output_file data/padded/dataset.hdf5 --max_len 1500
    
    # Entire directory
    python pad_dataset.py --input_dir data/merged --output_dir data/padded --max_len 1500
"""

import h5py
import numpy as np
import argparse
import os
import glob
from tqdm import tqdm

def get_episode_length(hdf5_source):
    """Get the length of an episode from qpos data.
    
    Args:
        hdf5_source: Either an h5py.File or an h5py.Group
    """
    # Check if this is a file with episode groups or a single episode file
    if 'observations/qpos' in hdf5_source:
        # Single episode file
        return hdf5_source['observations/qpos'].shape[0]
    elif 'episode_0' in hdf5_source:
        # File with episode groups
        return hdf5_source['episode_0']['observations/qpos'].shape[0]
    else:
        # Try to find any episode group
        episode_keys = [k for k in hdf5_source.keys() if k.startswith('episode_')]
        if episode_keys:
            return hdf5_source[episode_keys[0]]['observations/qpos'].shape[0]
    return 0

def find_max_length(hdf5_file):
    """Find the maximum episode length in the dataset."""
    max_len = 0
    
    # Check if this is a single episode file or multi-episode file
    if 'observations/qpos' in hdf5_file:
        # Single episode file
        return hdf5_file['observations/qpos'].shape[0]
    
    # Multi-episode file
    episode_keys = [k for k in hdf5_file.keys() if k.startswith('episode_')]
    
    for ep_key in tqdm(episode_keys, desc="Finding max length", leave=False):
        ep_len = get_episode_length(hdf5_file[ep_key])
        max_len = max(max_len, ep_len)
    
    return max_len

def pad_dataset(input_path, output_path, max_len=None, pad_value=0):
    """
    Pad all episodes to the same length.
    
    Args:
        input_path: Path to input HDF5 file
        output_path: Path to output HDF5 file
        max_len: Maximum length to pad to (if None, use dataset max)
        pad_value: Value to use for padding (default: 0)
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    with h5py.File(input_path, 'r') as in_f:
        # Determine if this is a single episode file or multi-episode file
        is_single_episode = 'observations/qpos' in in_f
        
        if is_single_episode:
            # Single episode file - pad the entire file
            if max_len is None:
                max_len = get_episode_length(in_f)
                print(f"Single episode file, length: {max_len}")
            
            ep_len = get_episode_length(in_f)
            
            with h5py.File(output_path, 'w') as out_f:
                # Pad each dataset
                for key in in_f.keys():
                    _pad_group_recursive(in_f, out_f, key, ep_len, max_len, pad_value)
                
                # Copy attributes
                for attr_key, attr_val in in_f.attrs.items():
                    out_f.attrs[attr_key] = attr_val
                
                # Add padding metadata
                out_f.attrs['original_length'] = ep_len
                out_f.attrs['padded_length'] = max_len
                out_f.attrs['padding_applied'] = (ep_len < max_len)
        else:
            # Multi-episode file
            episode_keys = sorted([k for k in in_f.keys() if k.startswith('episode_')])
            
            # Find max length if not specified
            if max_len is None:
                print("Finding maximum episode length...")
                max_len = find_max_length(in_f)
                print(f"Maximum episode length: {max_len}")
            
            # Create output file
            with h5py.File(output_path, 'w') as out_f:
                print(f"\nPadding {len(episode_keys)} episodes to length {max_len}...")
                
                for ep_key in tqdm(episode_keys, desc="Padding episodes"):
                    ep_group = in_f[ep_key]
                    ep_len = get_episode_length(ep_group)
                    
                    # Create output episode group
                    out_ep = out_f.create_group(ep_key)
                    
                    # Pad each dataset in the episode
                    for key in ep_group.keys():
                        _pad_group_recursive(ep_group, out_ep, key, ep_len, max_len, pad_value)
                    
                    # Copy attributes
                    for attr_key, attr_val in ep_group.attrs.items():
                        out_ep.attrs[attr_key] = attr_val
                    
                    # Add padding metadata
                    out_ep.attrs['original_length'] = ep_len
                    out_ep.attrs['padded_length'] = max_len
                    out_ep.attrs['padding_applied'] = (ep_len < max_len)
                
                # Copy global attributes
                for attr_key, attr_val in in_f.attrs.items():
                    out_f.attrs[attr_key] = attr_val
                
                # Add padding metadata
                out_f.attrs['max_episode_length'] = max_len
                out_f.attrs['padding_value'] = pad_value
    
    print(f"\n✅ Padded dataset saved to: {output_path}")
    print(f"   Episode length: {max_len}")
    print(f"   File size: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")

def _pad_group_recursive(src_group, dst_group, key, ep_len, max_len, pad_value):
    """Recursively pad datasets in a group."""
    item = src_group[key]
    
    if isinstance(item, h5py.Group):
        # Create subgroup and recurse
        sub_group = dst_group.create_group(key)
        for sub_key in item.keys():
            _pad_group_recursive(item, sub_group, sub_key, ep_len, max_len, pad_value)
        # Copy attributes
        for attr_key, attr_val in item.attrs.items():
            sub_group.attrs[attr_key] = attr_val
    else:
        # Dataset - pad if temporal
        data = item[:]
        
        if len(data.shape) == 0:
            # Scalar, copy as-is
            dst_group[key] = data
        elif data.shape[0] == ep_len:
            # Temporal data, needs padding
            pad_len = max_len - ep_len
            
            if pad_len > 0:
                # Use LAST VALUE PADDING instead of zero padding
                # This prevents unnatural arm poses (e.g., all joints at 0 radians)
                last_frame = data[-1:, ...]  # Keep dimensions: (1, ...)
                padding = np.repeat(last_frame, pad_len, axis=0)  # Repeat last frame
                padded_data = np.concatenate([data, padding], axis=0)
            else:
                padded_data = data
            
            # Save with chunking
            chunk_shape = (1,) + padded_data.shape[1:]
            dst_group.create_dataset(
                key,
                data=padded_data,
                chunks=chunk_shape,
                compression='gzip'
            )
        else:
            # Non-temporal data, copy as-is
            dst_group[key] = data

def verify_padding(hdf5_path):
    """Verify that all episodes have the same length."""
    with h5py.File(hdf5_path, 'r') as f:
        # Check if single episode file
        if 'observations/qpos' in f:
            ep_len = get_episode_length(f)
            print(f"\nVerification:")
            print(f"  Single episode file")
            print(f"  Episode length: {ep_len}")
            print(f"  ✅ File verified")
            return
        
        # Multi-episode file
        episode_keys = [k for k in f.keys() if k.startswith('episode_')]
        
        lengths = []
        for ep_key in episode_keys:
            ep_len = get_episode_length(f[ep_key])
            lengths.append(ep_len)
        
        unique_lengths = set(lengths)
        
        print(f"\nVerification:")
        print(f"  Total episodes: {len(episode_keys)}")
        print(f"  Unique lengths: {unique_lengths}")
        
        if not lengths:
            print(f"  ⚠️  No episodes found in file!")
        elif len(unique_lengths) == 1:
            print(f"  ✅ All episodes have uniform length: {list(unique_lengths)[0]}")
        else:
            print(f"  ❌ Episodes have varying lengths!")
            print(f"     Min: {min(lengths)}, Max: {max(lengths)}")

def process_directory(input_dir, output_dir, max_len=None, pad_value=0, verify=False):
    """
    Process all HDF5 files in a directory.
    
    Args:
        input_dir: Input directory containing HDF5 files
        output_dir: Output directory for padded files
        max_len: Maximum length to pad to (if None, auto-detect from all files)
        pad_value: Value to use for padding
        verify: Whether to verify output
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all HDF5 files
    hdf5_files = glob.glob(os.path.join(input_dir, '*.hdf5'))
    
    if not hdf5_files:
        print(f"No HDF5 files found in {input_dir}")
        return
    
    print(f"Found {len(hdf5_files)} HDF5 files in {input_dir}")
    
    # Auto-detect max length across all files if not specified
    if max_len is None:
        print("\nFinding maximum episode length across all files...")
        global_max_len = 0
        for hdf5_file in tqdm(hdf5_files, desc="Scanning files"):
            with h5py.File(hdf5_file, 'r') as f:
                file_max_len = find_max_length(f)
                global_max_len = max(global_max_len, file_max_len)
        max_len = global_max_len
        print(f"Global maximum episode length: {max_len}")
    
    # Process each file
    print(f"\nProcessing files with max_len={max_len}...")
    for hdf5_file in hdf5_files:
        filename = os.path.basename(hdf5_file)
        output_path = os.path.join(output_dir, filename)
        
        print(f"\n{'='*60}")
        print(f"Processing: {filename}")
        print(f"{'='*60}")
        
        pad_dataset(hdf5_file, output_path, max_len, pad_value)
        
        if verify:
            verify_padding(output_path)
    
    print(f"\n{'='*60}")
    print(f"✅ All files processed successfully!")
    print(f"   Output directory: {output_dir}")
    print(f"{'='*60}")

def main():
    parser = argparse.ArgumentParser(description='Pad HDF5 dataset episodes to uniform length')
    
    # Single file mode
    parser.add_argument('--input_file', type=str, default=None,
                        help='Path to input HDF5 file (single file mode)')
    parser.add_argument('--output_file', type=str, default=None,
                        help='Path to output HDF5 file (single file mode)')
    
    # Directory mode
    parser.add_argument('--input_dir', type=str, default=None,
                        help='Input directory containing HDF5 files (directory mode)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for padded files (directory mode)')
    
    # Common options
    parser.add_argument('--max_len', type=int, default=None,
                        help='Maximum length to pad to (default: auto-detect from dataset)')
    parser.add_argument('--pad_value', type=float, default=0.0,
                        help='Value to use for padding (default: 0)')
    parser.add_argument('--verify', action='store_true',
                        help='Verify output after padding')
    
    args = parser.parse_args()
    
    # Determine mode
    if args.input_dir and args.output_dir:
        # Directory mode
        process_directory(args.input_dir, args.output_dir, args.max_len, args.pad_value, args.verify)
    elif args.input_file and args.output_file:
        # Single file mode
        pad_dataset(args.input_file, args.output_file, args.max_len, args.pad_value)
        if args.verify:
            verify_padding(args.output_file)
    else:
        parser.error("Either (--input_file and --output_file) or (--input_dir and --output_dir) must be specified")

if __name__ == '__main__':
    main()
