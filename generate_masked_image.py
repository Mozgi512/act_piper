
import matplotlib.pyplot as plt
import numpy as np
from piper_sim_env import make_sim_env
import os
import cv2

def generate_masked_image():
    # Create env
    env = make_sim_env('sim_many_cubes')
    ts = env.reset()
    
    # Get top camera image
    cam_name = 'top'
    raw_image = ts.observation['images'][cam_name] # (480, 640, 3) uint8
    
    h, w, c = raw_image.shape
    
    # Parameters
    offset = 40
    start = w//2 - offset
    end = w - offset
    
    # Crop
    cropped_image = raw_image[:, start:end, :].copy() 
    # Shape: (480, 320, 3)
    
    # The strip to process is the leftmost 'offset' pixels of the crop
    strip_width = offset
    strip = cropped_image[:, :strip_width, :]
    
    # Define thresholds for Red, Green, Blue
    # Assuming RGB format
    # Sim colors are pure: [255, 0, 0], [0, 255, 0], [0, 0, 255] roughly
    # We use a threshold to capture them even with shading
    
    lower_red = np.array([100, 0, 0])
    upper_red = np.array([255, 100, 100])
    
    lower_green = np.array([0, 100, 0])
    upper_green = np.array([100, 255, 100])
    
    lower_blue = np.array([0, 0, 150])
    upper_blue = np.array([100, 100, 255])
    
    # Create masks
    mask_r = cv2.inRange(strip, lower_red, upper_red)
    mask_g = cv2.inRange(strip, lower_green, upper_green)
    mask_b = cv2.inRange(strip, lower_blue, upper_blue)
    
    # Combine masks
    combined_mask = cv2.bitwise_or(mask_r, mask_g)
    combined_mask = cv2.bitwise_or(combined_mask, mask_b)
    
    # Apply mask to strip (black out non-matching pixels)
    masked_strip = cv2.bitwise_and(strip, strip, mask=combined_mask)
    
    # Put back the masked strip into the cropped image
    cropped_image[:, :strip_width, :] = masked_strip
    
    # Save images
    save_dir = "/home/act/act_piper"
    plt.imsave(os.path.join(save_dir, "masked_right_arm.png"), cropped_image)
    print(f"Saved masked image to {save_dir}/masked_right_arm.png")

if __name__ == '__main__':
    generate_masked_image()
