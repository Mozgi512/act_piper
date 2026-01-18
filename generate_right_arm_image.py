
import matplotlib.pyplot as plt
import numpy as np
from piper_sim_env import make_sim_env
from einops import rearrange
import os

def generate_images():
    # Create env
    env = make_sim_env('sim_many_cubes')
    ts = env.reset()
    
    # Get top camera image
    # Shape is (480, 640, 3) probably
    cam_name = 'top'
    raw_image = ts.observation['images'][cam_name]
    print(f"Original image shape: {raw_image.shape}")
    
    h, w, c = raw_image.shape
    
    # Original crop (Right arm)
    # Right half: [w//2:]
    original_crop = raw_image[:, w//2:, :]
    print(f"Original crop shape: {original_crop.shape}")
    
    # New crop (Shift left by 40 pixels)
    offset = 40
    # Range: [w//2 - offset : w - offset]
    start = w//2 - offset
    end = w - offset
    new_crop = raw_image[:, start:end, :]
    print(f"New crop shape: {new_crop.shape}")
    
    # Save images
    save_dir = "/home/act/act_piper"
    plt.imsave(os.path.join(save_dir, "right_arm_original.png"), original_crop)
    plt.imsave(os.path.join(save_dir, "right_arm_shifted.png"), new_crop)
    print(f"Saved images to {save_dir}")

if __name__ == '__main__':
    generate_images()
