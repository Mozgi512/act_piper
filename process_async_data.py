import os
import h5py
import numpy as np
import cv2
import argparse
import time


def apply_rgb_mask_to_strip(image, strip_width=40):
    """
    Applies a color mask to the leftmost `strip_width` pixels of the image.
    Preserves Red, Green, and Blue colors; blacks out everything else.
    Image is expected to be (H, W, 3) numpy array (uint8).
    """
    if image.shape[1] < strip_width:
        return image
        
    strip = image[:, :strip_width, :]
    
    lower_red = np.array([100, 0, 0], dtype=np.uint8)
    upper_red = np.array([255, 100, 100], dtype=np.uint8)
    
    lower_green = np.array([0, 100, 0], dtype=np.uint8)
    upper_green = np.array([100, 255, 100], dtype=np.uint8)
    
    # Floor blue max is ~102, so use 150 to be safe
    lower_blue = np.array([0, 0, 150], dtype=np.uint8)
    upper_blue = np.array([100, 100, 255], dtype=np.uint8)
    
    mask_r = cv2.inRange(strip, lower_red, upper_red)
    mask_g = cv2.inRange(strip, lower_green, upper_green)
    mask_b = cv2.inRange(strip, lower_blue, upper_blue)
    
    combined_mask = cv2.bitwise_or(mask_r, mask_g)
    combined_mask = cv2.bitwise_or(combined_mask, mask_b)
    
    masked_strip = cv2.bitwise_and(strip, strip, mask=combined_mask)
    
    image[:, :strip_width, :] = masked_strip
    return image


def apply_rgb_mask_to_right_strip(image, strip_width=40):
    """
    Applies a color mask to the rightmost `strip_width` pixels of the image.
    Preserves Red, Green, and Blue colors; blacks out everything else.
    Image is expected to be (H, W, 3) numpy array (uint8).
    """
    if image.shape[1] < strip_width:
        return image
        
    strip = image[:, -strip_width:, :]
    
    lower_red = np.array([100, 0, 0], dtype=np.uint8)
    upper_red = np.array([255, 100, 100], dtype=np.uint8)
    
    lower_green = np.array([0, 100, 0], dtype=np.uint8)
    upper_green = np.array([100, 255, 100], dtype=np.uint8)
    
    # Floor blue max is ~102, so use 150 to be safe
    lower_blue = np.array([0, 0, 150], dtype=np.uint8)
    upper_blue = np.array([100, 100, 255], dtype=np.uint8)
    
    mask_r = cv2.inRange(strip, lower_red, upper_red)
    mask_g = cv2.inRange(strip, lower_green, upper_green)
    mask_b = cv2.inRange(strip, lower_blue, upper_blue)
    
    combined_mask = cv2.bitwise_or(mask_r, mask_g)
    combined_mask = cv2.bitwise_or(combined_mask, mask_b)
    
    masked_strip = cv2.bitwise_and(strip, strip, mask=combined_mask)
    
    image[:, -strip_width:, :] = masked_strip
    return image


def process_segment(qpos, qvel, action, images, camera_names, is_left, img_w):
    """
    Extract single-arm data and process images.
    
    Args:
        is_left: If True, extract left arm data (indices 0-6). Otherwise right (7-13).
        img_w: Image width for scaling
    """
    # Define constants based on resolution
    scale = img_w / 640.0
    SPLIT_COL = int(320 * scale)
    MASK_WIDTH = int(40 * scale)
    
    # Extract single arm joint data
    if is_left:
        seg_qpos = qpos[:, :7]
        seg_qvel = qvel[:, :7]
        seg_action = action[:, :7]
    else:
        seg_qpos = qpos[:, 7:14]
        seg_qvel = qvel[:, 7:14]
        seg_action = action[:, 7:14]
    
    # Process images
    seg_images = {}
    for cam_name, img_seq in images.items():
        if is_left:
            # Crop Left Half: [:, :SPLIT_COL, :]
            cropped = img_seq[:, :, :SPLIT_COL, :]
            # Apply Mask to Right Edge
            masked_seq = []
            for i in range(cropped.shape[0]):
                frame = cropped[i].copy()
                frame = apply_rgb_mask_to_right_strip(frame, strip_width=MASK_WIDTH)
                masked_seq.append(frame)
            seg_images[cam_name] = np.array(masked_seq)
        else:
            # Crop Right Half: [:, SPLIT_COL:, :]
            cropped = img_seq[:, :, SPLIT_COL:, :]
            # Apply Mask to Left Edge
            masked_seq = []
            for i in range(cropped.shape[0]):
                frame = cropped[i].copy()
                frame = apply_rgb_mask_to_strip(frame, strip_width=MASK_WIDTH)
                masked_seq.append(frame)
            seg_images[cam_name] = np.array(masked_seq)
    
    return seg_qpos, seg_qvel, seg_action, seg_images


def save_hdf5(dataset_path, qpos, qvel, action, images, camera_names):
    """Save segment to HDF5 file WITHOUT metadata group."""
    max_timesteps = qpos.shape[0]
    
    # Get actual image dimensions from processed images
    sample_img = list(images.values())[0]
    img_height, img_width = sample_img.shape[1:3]  # (T, H, W, C)
    
    with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024**2*2) as root:
        root.attrs['sim'] = True
        root.attrs['compress'] = True
        
        obs = root.create_group('observations')
        image_group = obs.create_group('images')
        
        for cam_name in camera_names:
            _ = image_group.create_dataset(
                cam_name, 
                (max_timesteps, img_height, img_width, 3), 
                dtype='uint8',
                chunks=(1, img_height, img_width, 3),
            )
        
        # For independent segments: 7-dim actions
        action_dim = action.shape[1]
        _ = obs.create_dataset('qpos', (max_timesteps, action_dim))
        _ = obs.create_dataset('qvel', (max_timesteps, action_dim))
        _ = root.create_dataset('action', (max_timesteps, action_dim))
        
        # Write data
        for cam_name in camera_names:
            root[f'/observations/images/{cam_name}'][...] = images[cam_name]
        
        root['/observations/qpos'][...] = qpos
        root['/observations/qvel'][...] = qvel
        root['/action'][...] = action
        
        # NOTE: Metadata group is intentionally NOT included in processed files


def process_episode(episode_idx, dataset_dir, camera_names, output_dirs):
    """
    Process one episode: split into segments based on metadata.
    
    Args:
        output_dirs: {'left_independent': path, 'right_independent': path, 
                     'cooperative_assembly': path, 'cooperative_place': path}
    """
    dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
    
    if not os.path.exists(dataset_path):
        print(f"Episode {episode_idx} not found at {dataset_path}")
        return 0
    
    with h5py.File(dataset_path, 'r') as root:
        qpos = root['/observations/qpos'][()]
        qvel = root['/observations/qvel'][()]
        action = root['/action'][()]
        
        images = {}
        for cam_name in camera_names:
            images[cam_name] = root[f'/observations/images/{cam_name}'][()]
        
        # Load metadata
        if 'metadata' not in root or 'left_segments' not in root['metadata']:
            print(f"Episode {episode_idx}: No metadata found, skipping")
            return 0
        
        left_segments = root['metadata/left_segments'][()]
        right_segments = root['metadata/right_segments'][()]
    
    # Detect image resolution
    sample_img = list(images.values())[0]
    img_h, img_w, _ = sample_img.shape[1:]
    
    # 1. Merge and group metadata
    all_raw_segs = []
    for seg in left_segments:
        s = {'start': int(seg['start']), 'end': int(seg['end']), 
             'type': seg['type'].decode('utf-8') if isinstance(seg['type'], bytes) else seg['type'],
             'top_arm': seg['top_arm'].decode('utf-8') if seg['top_arm'] else '',
             'coop_split': int(seg['coop_split']), 'arm': 'left'}
        all_raw_segs.append(s)
    for seg in right_segments:
        s = {'start': int(seg['start']), 'end': int(seg['end']), 
             'type': seg['type'].decode('utf-8') if isinstance(seg['type'], bytes) else seg['type'],
             'top_arm': seg['top_arm'].decode('utf-8') if seg['top_arm'] else '',
             'coop_split': int(seg['coop_split']), 'arm': 'right'}
        all_raw_segs.append(s)

    # Group cooperative tasks by start time, keep independent tasks separate
    merged_segs = []
    coop_groups = {} # start -> list of segments
    
    for s in all_raw_segs:
        if s['type'] == 'cooperative':
            if s['start'] not in coop_groups:
                coop_groups[s['start']] = []
            coop_groups[s['start']].append(s)
        else:
            merged_segs.append(s)
            
    for start, seg_list in coop_groups.items():
        # Merge cooperative info: pick MAX end and MAX coop_split
        max_end = max(s['end'] for s in seg_list)
        max_split = max(s['coop_split'] for s in seg_list)
        # Find the top_arm (should be same in both, but pick first non-empty)
        top_arm = next((s['top_arm'] for s in seg_list if s['top_arm']), '')
        
        merged_segs.append({
            'type': 'cooperative',
            'start': start,
            'end': max_end,
            'coop_split': max_split,
            'top_arm': top_arm,
            'arm': 'both'
        })
    
    # Sort merged segments by start time for consistent episode indexing
    merged_segs.sort(key=lambda x: x['start'])
    
    segment_count = 0
    for seg in merged_segs:
        start, end, seg_type = seg['start'], seg['end'], seg['type']
        top_arm = seg['top_arm']
        coop_split = seg['coop_split']
        arm = seg['arm']
        
        if seg_type == 'independent':
            # Independent: [start-10 : end+10]
            adj_start = max(0, start - 10)
            adj_end = min(len(qpos), end + 10)
            is_left = (arm == 'left')
            
            seg_qpos, seg_qvel, seg_action, seg_images = process_segment(
                qpos[adj_start:adj_end],
                qvel[adj_start:adj_end],
                action[adj_start:adj_end],
                {k: v[adj_start:adj_end] for k, v in images.items()},
                camera_names,
                is_left=is_left,
                img_w=img_w
            )
            
            out_dir = output_dirs['left_independent'] if is_left else output_dirs['right_independent']
            seg_folder_name = f'seg_{segment_count}'
            seg_full_dir = os.path.join(out_dir, seg_folder_name)
            os.makedirs(seg_full_dir, exist_ok=True)
            output_path = os.path.join(seg_full_dir, f'episode_{episode_idx}')
            
            save_hdf5(output_path, seg_qpos, seg_qvel, seg_action, seg_images, camera_names)
            segment_count += 1
            
        elif seg_type == 'cooperative':
            # Cooperative task: split into assembly + placement
            
            # Assembly: [start : coop_split+10] (dual-arm until Top returns home)
            if coop_split > 0:
                adj_split_end = min(len(qpos), coop_split + 10)
                
                # Assembly is always 14-dim (both arms)
                seg_qpos = qpos[start:adj_split_end]
                seg_qvel = qvel[start:adj_split_end]
                seg_action = action[start:adj_split_end]
                seg_images = {k: v[start:adj_split_end] for k, v in images.items()}
                
                seg_folder_name = f'seg_{segment_count}'
                seg_full_dir = os.path.join(output_dirs['cooperative_assembly'], seg_folder_name)
                os.makedirs(seg_full_dir, exist_ok=True)
                output_path = os.path.join(seg_full_dir, f'episode_{episode_idx}')
                
                save_hdf5(output_path, seg_qpos, seg_qvel, seg_action, seg_images, camera_names)
                segment_count += 1
                
                # Placement: [coop_split+10 : end+10] (single-arm Base only)
                # Determine which arm is base (opposite of top)
                base_is_left = (top_arm == 'right')  # If top is right, base is left
                
                adj_place_start = min(len(qpos), coop_split + 10)
                adj_place_end = min(len(qpos), end + 10)
                
                if adj_place_start < adj_place_end:
                    seg_qpos, seg_qvel, seg_action, seg_images = process_segment(
                        qpos[adj_place_start:adj_place_end],
                        qvel[adj_place_start:adj_place_end],
                        action[adj_place_start:adj_place_end],
                        {k: v[adj_place_start:adj_place_end] for k, v in images.items()},
                        camera_names,
                        is_left=base_is_left,
                        img_w=img_w
                    )
                    
                    # Save to appropriate base directory
                    output_dir = output_dirs['left_independent'] if base_is_left else output_dirs['right_independent']
                    
                    seg_folder_name = f'seg_{segment_count}'
                    seg_full_dir = os.path.join(output_dir, seg_folder_name)
                    os.makedirs(seg_full_dir, exist_ok=True)
                    output_path = os.path.join(seg_full_dir, f'episode_{episode_idx}')

                    save_hdf5(output_path, seg_qpos, seg_qvel, seg_action, seg_images, camera_names)
                    segment_count += 1
    
    return segment_count
    
    return segment_count


def main(args):
    dataset_dir = args['dataset_dir']
    num_episodes = args['num_episodes']
    
    # Output directories
    output_dirs = {
        'left_independent': dataset_dir + '_left_independent',
        'right_independent': dataset_dir + '_right_independent',
        'cooperative_assembly': dataset_dir + '_cooperative_assembly',
    }
    
    print(f"Processing async data from {dataset_dir}")
    print(f"Output directories:")
    for key, path in output_dirs.items():
        print(f"  {key}: {path}")
    
    # Inspect first episode for camera names
    first_ep_path = os.path.join(dataset_dir, 'episode_0.hdf5')
    if not os.path.exists(first_ep_path):
        print(f"Error: Could not find {first_ep_path}")
        return
    
    with h5py.File(first_ep_path, 'r') as f:
        camera_names = list(f['/observations/images'].keys())
    
    total_segments = 0
    t0 = time.time()
    
    for i in range(num_episodes):
        seg_count = process_episode(i, dataset_dir, camera_names, output_dirs)
        total_segments += seg_count
        if (i+1) % 10 == 0:
            print(f"Processed {i+1}/{num_episodes} episodes...")
    
    print(f"Finished processing {num_episodes} episodes -> {total_segments} segments in {time.time() - t0:.1f} seconds.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', action='store', type=str, required=True,
                       help='Directory containing raw episodes with metadata')
    parser.add_argument('--num_episodes', action='store', type=int, required=True,
                       help='Number of episodes to process')
    
    main(vars(parser.parse_args()))
