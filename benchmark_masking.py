
import time
import numpy as np
import cv2
from utils import apply_rgb_mask_to_strip

def benchmark():
    # Create dummy image (480, 640, 3)
    dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Warmup
    for _ in range(10):
        apply_rgb_mask_to_strip(dummy_image.copy(), strip_width=40)
        
    # Benchmark
    iterations = 1000
    start_time = time.time()
    for _ in range(iterations):
        # working on a view slice in reality, but copy here to simulate worst case or fresh data
        # In policy_switcher it is passed as a slice or array.
        # Let's pass the slice as in the real code
        # In real code: curr_image is passed.
        # But wait, the function takes the whole image and slices internally.
        apply_rgb_mask_to_strip(dummy_image, strip_width=40)
        
    end_time = time.time()
    avg_time = (end_time - start_time) / iterations
    print(f"Average time per call: {avg_time*1000:.4f} ms")

if __name__ == '__main__':
    benchmark()
