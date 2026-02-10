
import cv2
import numpy as np
from utils import apply_policy_mask

def main():
    img_path = 'test.png'
    print(f"Loading {img_path}...")
    
    # Read Image (BGR)
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"Error: Could not read {img_path}")
        return

    # Convert to RGB (utils expects RGB)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # 1. Apply INDEP Mask (Mask Green & Blue > 240)
    print("Applying INDEP Mask (Masking Green & Blue)...")
    img_indep = apply_policy_mask(img_rgb, 'INDEP')
    
    # 2. Apply COOP Mask (Mask Red > 240)
    print("Applying COOP Mask (Masking Red)...")
    img_coop = apply_policy_mask(img_rgb, 'COOP') 
    # Note: apply_policy_mask uses .copy(), but since we passed img_rgb again, it's fine.
    # Wait, img_rgb is modified in place if not copied inside function?
    # utils.py: "image = image.copy()" -> Yes it copies.
    
    # Save Results (Convert back to BGR for cv2.imwrite)
    img_indep_bgr = cv2.cvtColor(img_indep, cv2.COLOR_RGB2BGR)
    cv2.imwrite('test_indep_masked.png', img_indep_bgr)
    print("Saved test_indep_masked.png")
    
    img_coop_bgr = cv2.cvtColor(img_coop, cv2.COLOR_RGB2BGR)
    cv2.imwrite('test_coop_masked.png', img_coop_bgr)
    print("Saved test_coop_masked.png")

if __name__ == '__main__':
    main()
