import h5py
import numpy as np
import os
import glob
import argparse

def inspect(dataset_dir, num_episodes=5):
    print(f"Inspecting dataset: {dataset_dir}")
    files = sorted(glob.glob(os.path.join(dataset_dir, '*.hdf5')))
    if not files:
        print("No .hdf5 files found!")
        return

    print(f"Found {len(files)} files. Inspecting first {num_episodes}...")
    
    for i, fpath in enumerate(files[:num_episodes]):
        print(f"\n--- Episode: {os.path.basename(fpath)} ---")
        try:
            with h5py.File(fpath, 'r') as root:
                qpos = root['/observations/qpos'][()]
                action = root['/action'][()]
                
                print(f"qpos shape: {qpos.shape}")
                print(f"action shape: {action.shape}")
                
                qpos_mean = np.mean(qpos, axis=0)
                qpos_std = np.std(qpos, axis=0)
                action_mean = np.mean(action, axis=0)
                action_std = np.std(action, axis=0)
                
                print(f"qpos mean: {qpos_mean}")
                print(f"qpos std:  {qpos_std}")
                print(f"action mean: {action_mean}")
                print(f"action std:  {action_std}")
                
                # Check for dynamic content
                is_static = np.all(action_std < 1e-4)
                if is_static:
                    print("WARNING: Data appears STATIC (std < 1e-4)")
                else:
                    print("Data appears DYNAMIC")

        except Exception as e:
            print(f"Error reading {fpath}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir', help='Path to dataset directory')
    args = parser.parse_args()
    inspect(args.dataset_dir)
