
import h5py
import os
import glob
import numpy as np

def check_dataset_lengths(dataset_dir):
    files = glob.glob(os.path.join(dataset_dir, '*.hdf5'))
    if not files:
        print(f"No files in {dataset_dir}")
        return

    print(f"Checking {len(files)} files in {dataset_dir}...")
    lengths = []
    for fpath in files:
        with h5py.File(fpath, 'r') as f:
            if 'observations/qpos' in f:
                l = f['observations/qpos'].shape[0]
                lengths.append(l)
            elif 'episode_0' in f:
                l = f['episode_0/observations/qpos'].shape[0]
                lengths.append(l)
    
    if len(set(lengths)) == 1:
        print(f"Uniform length: {lengths[0]}")
    else:
        print(f"Varying lengths: {min(lengths)} - {max(lengths)}")

if __name__ == "__main__":
    check_dataset_lengths('data/ICTICT/_cooperative_assembly/training')
    check_dataset_lengths('data/ICTICT/_left_independent/training')
