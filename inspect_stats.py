import pickle
import torch
import numpy as np
import sys
import os

def inspect_stats(ckpt_dir):
    stats_path = os.path.join(ckpt_dir, 'dataset_stats.pkl')
    print(f"Loading stats from: {stats_path}")
    
    if not os.path.exists(stats_path):
        print("File not found!")
        return

    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)
    
    for k, v in stats.items():
        print(f"\nKey: {k}")
        if isinstance(v, (np.ndarray, torch.Tensor)):
            print(f"Shape: {v.shape}")
            print(f"Value:\n{v}")
        else:
            print(f"Value: {v}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_stats.py <ckpt_dir>")
    else:
        inspect_stats(sys.argv[1])
