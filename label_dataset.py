import h5py
import cv2
import numpy as np
import os
import argparse
from tqdm import tqdm

def detect_objects(image):
    """
    Detects R, G, B objects in the image using HSV color masking.
    Returns a list of detected objects with color, bounding box, and center.
    """
    # image is RGB (from HDF5) -> Convert to HSV for robust detection
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # Define color ranges in HSV
    # Balanced Mode: Relaxed Saturation to catch desaturated/shadowed cubes,
    # but maintaining narrow Blue Hue to exclude background pillars.
    ranges = {
        'Red': [(np.array([0, 80, 40]), np.array([10, 255, 255])),
                (np.array([170, 80, 40]), np.array([180, 255, 255]))],
        'Green': [(np.array([35, 80, 40]), np.array([85, 255, 255]))],
        'Blue': [(np.array([118, 80, 40]), np.array([125, 255, 255]))]
    }
    
    detected = []
    for color_name, color_ranges in ranges.items():
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for low, high in color_ranges:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, low, high))
            
        # Clean up mask
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 60: # Filter small noise
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = w / float(h)
                
                # Cubes should be roughly square (aspect ratio near 1.0)
                # Allowing wider range (0.3 to 3.0) for partially hidden cubes.
                is_cube_shape = 0.3 < aspect_ratio < 3.0
                
                if is_cube_shape:
                    detected.append({
                        'color': color_name,
                        'bbox': (x, y, w, h),
                        'center': (x + w//2, y + h//2),
                        'area': area
                    })
    return detected

def main():
    parser = argparse.ArgumentParser(description='Label R/G/B objects in HDF5 dataset images.')
    parser.add_argument('--input_path', type=str, required=True, help='Path to hdf5 file or directory')
    parser.add_argument('--output_dir', type=str, default='labeled_output', help='Directory to save results')
    parser.add_argument('--cam_name', type=str, default='top', help='Camera name in HDF5')
    parser.add_argument('--max_frames', type=int, default=10, help='Max frames to process per HDF5 (for sampling)')
    args = parser.parse_args()

    # Handle single file or directory
    if os.path.isdir(args.input_path):
        hdf5_files = [os.path.join(args.input_path, f) for f in os.listdir(args.input_path) if f.endswith('.hdf5')]
    else:
        hdf5_files = [args.input_path]

    os.makedirs(args.output_dir, exist_ok=True)

    for hdf5_path in hdf5_files:
        print(f"Processing: {hdf5_path}")
        episode_name = os.path.splitext(os.path.basename(hdf5_path))[0]
        episode_out_dir = os.path.join(args.output_dir, episode_name)
        os.makedirs(episode_out_dir, exist_ok=True)

        try:
            with h5py.File(hdf5_path, 'r') as root:
                images = root[f'observations/images/{args.cam_name}'][:]
        except Exception as e:
            print(f"Error reading {hdf5_path}: {e}")
            continue

        num_frames = min(len(images), args.max_frames)
        # Sample frames evenly if max_frames < len(images)
        indices = np.linspace(0, len(images)-1, num_frames).astype(int)

        for i in indices:
            img = images[i]
            H, W, _ = img.shape
            
            # 1. Crop: Upper middle section (2nd quartile)
            # User wants "2nd from top" -> [H/4, H/2]
            # No horizontal trimming to avoid information loss
            y_start, y_end = H//4, H//2
            crop = img[y_start:y_end, :]
            
            # 2. Detect R, G, B objects
            objects = detect_objects(crop)
            
            # 3. Sort Symmetrically from Center
            mid_x = W // 2
            left_side = [obj for obj in objects if obj['center'][0] < mid_x]
            right_side = [obj for obj in objects if obj['center'][0] >= mid_x]
            
            # Left: Sort by X descending (closer to center is 0)
            left_side.sort(key=lambda o: o['center'][0], reverse=True)
            # Right: Sort by X ascending (closer to center is 0)
            right_side.sort(key=lambda o: o['center'][0])
            
            # 4. Visualization
            # Convert RGB (sim) to BGR (opencv)
            viz = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
            
            # Label Colors for Visualization
            color_map = {'Red': (0, 0, 255), 'Green': (0, 255, 0), 'Blue': (255, 0, 0)}

            for idx, obj in enumerate(left_side):
                x, y, w, h = obj['bbox']
                color = color_map.get(obj['color'], (255, 255, 255))
                cv2.rectangle(viz, (x, y), (x + w, y + h), color, 2)
                # Label: L[idx] (e.g. L0, L1)
                cv2.putText(viz, f"L{idx}:{obj['color'][0]}", (x, y-5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            for idx, obj in enumerate(right_side):
                x, y, w, h = obj['bbox']
                color = color_map.get(obj['color'], (255, 255, 255))
                cv2.rectangle(viz, (x, y), (x + w, y + h), color, 2)
                # Label: R[idx] (e.g. R0, R1)
                cv2.putText(viz, f"R{idx}:{obj['color'][0]}", (x, y-5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Draw center line
            cv2.line(viz, (mid_x, 0), (mid_x, y_end-y_start), (255, 255, 255), 1)

            out_fn = f"frame_{i:04d}_labeled.png"
            cv2.imwrite(os.path.join(episode_out_dir, out_fn), viz)

    print(f"\nDone. Labeled images saved to: {args.output_dir}")

if __name__ == '__main__':
    main()
