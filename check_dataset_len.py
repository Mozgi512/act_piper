
import h5py
import os

base_dir = '/home/act/act_piper/data/variable_coop_dataset'
subdirs = ['BGR', 'BRG', 'GBR', 'GRB', 'RBG', 'RGB']

print(f"{'Task Type':<10} | {'Episode Len':<15}")
print("-" * 30)

for subdir in subdirs:
    file_path = os.path.join(base_dir, subdir, 'episode_0.hdf5')
    if os.path.exists(file_path):
        with h5py.File(file_path, 'r') as f:
            length = f['/observations/qpos'].shape[0]
            print(f"{subdir:<10} | {length:<15}")
    else:
        print(f"{subdir:<10} | {'Not Found':<15}")
