import h5py
import numpy as np

file_path = '/home/act/act_piper/piper_scripted_dataset/sim_four_objects/episode_2.hdf5'

try:
    with h5py.File(file_path, 'r') as root:
        print("Keys:", list(root.keys()))
        if 'observations' in root:
            obs = root['observations']
            if 'images' in obs:
                imgs = obs['images']
                print("Image keys:", list(imgs.keys()))
                for cam in imgs.keys():
                    shape = imgs[cam].shape
                    print(f"Camera '{cam}' shape: {shape}")
            else:
                print("No images in observations")
        else:
            print("No observations group")
except Exception as e:
    print(f"Error: {e}")
