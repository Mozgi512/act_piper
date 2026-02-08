import h5py
import numpy as np
import cv2
import os
import argparse
from utils import apply_rgb_mask_to_strip, apply_rgb_mask_to_right_strip

def visualize_final(dataset_dir):
    files = sorted([f for f in os.listdir(dataset_dir) if f.endswith('.hdf5')])
    if not files:
        print("No HDF5 files found")
        return

    fpath = os.path.join(dataset_dir, files[0])
    print(f"Reading {fpath}")
    
    with h5py.File(fpath, 'r') as root:
        img = root['/observations/images/top'][0] # H W C (240x160x3 Likely for independent, but let's assume raw input 480x640 if possible? No, dataset is 240x160)
        # Wait, if dataset is ALREADY cropped (240x160), we can't test "Left Half" logic if "Left Half" assumes 640 width.
        # But `independent_imitate_episodes.py` logic says:
        # h, w = shape
        # Left: :w//2
        # If input is 160w, Left gets 80w.
        
        # User said "Original 240x320 image".
        # But our inspections showed 240x160.
        # This implies `_right_independent` ALREADY contains only the half?
        # If so, `get_image` logic `curr_image[:, :, w//2:]` would split 160->80.
        # AND user said "Left Half of Original... Right 40px mask".
        
        # Let's verify what the "Original" is.
        # If the input to `get_image` (from sim) is 640w.
        # Then `w//2` is 320.
        # So "Left Half" is 320w.
        # "Right 40px mask" -> 280-320 masked.
        
        # But the dataset image I inspected was 160w.
        # This means the dataset is NOT 640w.
        # If I use the dataset image (160w) as "Sim Input" proxy, it's wrong.
        # I need a sample 640w image to demonstrate the logic correctly for the USER.
        # I can try to find a 640w image, or Resize the dataset image to 640w to simulate "Raw Sim Input".
        
        h, w, c = img.shape
        print(f"Loaded Sample Image: {h}x{w}x{c}")
        
        # Mocking a full-width sim image (Double width)
        # Create a 480x640 proxy by resizing (just for viz)
        # Or just use the logic on 160w to show "What happens"
        # User wants "Left and Right application example".
        # I should demonstrate on a fake 640w image to show the cropping.
        
        # Create Dummy 320w Image (User Correction)
        # 320x240
        dummy_img = np.zeros((240, 320, 3), dtype=np.uint8)
        
        # Left Half (0-160): Dark Gray (50) - Should be masked
        cv2.rectangle(dummy_img, (0, 0), (160, 240), (50, 50, 50), -1) 
        # Right Half (160-320): Dark Gray (50) - Should be masked
        cv2.rectangle(dummy_img, (160, 0), (320, 240), (50, 50, 50), -1) 
        
        # Add RGB Boxes in the Overlap Region (Center +/- 20px -> 140 to 180)
        # This tests if the mask preserves these colors.
        
        # Red Box (Top)
        cv2.rectangle(dummy_img, (145, 50), (175, 80), (0, 0, 255), -1) # BGR: Red
        # Green Box (Middle)
        cv2.rectangle(dummy_img, (145, 100), (175, 130), (0, 255, 0), -1) # BGR: Green
        # Blue Box (Bottom)
        cv2.rectangle(dummy_img, (145, 150), (175, 180), (255, 0, 0), -1) # BGR: Blue
        
        # Add text
        cv2.putText(dummy_img, "L", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        cv2.putText(dummy_img, "R", (210, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        
        cv2.imwrite('viz_input_320w_rgb.png', dummy_img)
        print("Created viz_input_320w_rgb.png (Simulated Sim Input with RGB Boxes)")
        
        # Convert to RGB for Logic
        dummy_img_rgb = cv2.cvtColor(dummy_img, cv2.COLOR_BGR2RGB)
        
        # --- Apply Left Arm Logic ---
        # Logic: Left Half (:160), Mask Right Strip (40px)
        w = 320
        left_arm_img = dummy_img_rgb[:, :w//2, :].copy() # 160w
        print(f"Left Crop Size: {left_arm_img.shape}")
        
        # Mask Right 40px of this 160w image
        left_arm_masked = apply_rgb_mask_to_right_strip(left_arm_img.copy(), strip_width=40)
        # Convert back to BGR for saving
        cv2.imwrite('viz_result_left_arm_320.png', cv2.cvtColor(left_arm_masked, cv2.COLOR_RGB2BGR))
        
        # --- Apply Right Arm Logic ---
        # Logic: Right Half (160:), Mask Left Strip (40px)
        right_arm_img = dummy_img_rgb[:, w//2:, :].copy() # 160w
        print(f"Right Crop Size: {right_arm_img.shape}")
        
        # Mask Left 40px of this 160w image
        right_arm_masked = apply_rgb_mask_to_strip(right_arm_img.copy(), strip_width=40)
        cv2.imwrite('viz_result_right_arm_320.png', cv2.cvtColor(right_arm_masked, cv2.COLOR_RGB2BGR))
        
        print("Saved viz_result_left_arm_320.png and viz_result_right_arm_320.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir')
    args = parser.parse_args()
    visualize_final(args.dataset_dir)
