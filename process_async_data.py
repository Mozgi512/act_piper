import os
import h5py
import numpy as np
import argparse
import time
from utils import apply_rgb_mask_to_strip, apply_rgb_mask_to_right_strip


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
    """Save segment to HDF5 file."""
    max_timesteps = qpos.shape[0]
    
    with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024**2*2) as root:
        root.attrs['sim'] = True
        root.attrs['compress'] = True
        
        obs = root.create_group('observations')
        image_group = obs.create_group('images')
        
        for cam_name in camera_names:
            _ = image_group.create_dataset(
                cam_name, 
                (max_timesteps, 240, 320, 3), 
                dtype='uint8',
                chunks=(1, 240, 320, 3),
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


def process_episode(episode_idx, dataset_dir, camera_names, output_dirs):
    """
    Process one episode: split into segments based on metadata.
    
    Args:
        output_dirs: {'left_independent': path, 'right_independent': path, 'cooperative': path}
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
    
    segment_count = 0
    
    # Process left independent segments
    for seg in left_segments:
        start, end, seg_type = int(seg['start']), int(seg['end']), seg['type'].decode('utf-8')
        
        if seg_type == 'independent':
            seg_qpos, seg_qvel, seg_action, seg_images = process_segment(
                qpos[start:end],
                qvel[start:end],
                action[start:end],
                {k: v[start:end] for k, v in images.items()},
                camera_names,
                is_left=True,
                img_w=img_w
            )
            
            output_path = os.path.join(output_dirs['left_independent'], f'episode_{episode_idx}_seg_{segment_count}')
            os.makedirs(output_dirs['left_independent'], exist_ok=True)
            save_hdf5(output_path, seg_qpos, seg_qvel, seg_action, seg_images, camera_names)
            segment_count += 1
    
    # Process right independent segments
    for seg in right_segments:
        start, end, seg_type = int(seg['start']), int(seg['end']), seg['type'].decode('utf-8')
        
        if seg_type == 'independent':
            seg_qpos, seg_qvel, seg_action, seg_images = process_segment(
                qpos[start:end],
                qvel[start:end],
                action[start:end],
                {k: v[start:end] for k, v in images.items()},
                camera_names,
                is_left=False,
                img_w=img_w
            )
            
            output_path = os.path.join(output_dirs['right_independent'], f'episode_{episode_idx}_seg_{segment_count}')
            os.makedirs(output_dirs['right_independent'], exist_ok=True)
            save_hdf5(output_path, seg_qpos, seg_qvel, seg_action, seg_images, camera_names)
            segment_count += 1
    
    # Process cooperative segments (from left_segments, they should match right_segments)
    for seg in left_segments:
        start, end, seg_type = int(seg['start']), int(seg['end']), seg['type'].decode('utf-8')
        
        if seg_type == 'cooperative':
            # For cooperative: keep full 14-dim data
            seg_qpos = qpos[start:end]
            seg_qvel = qvel[start:end]
            seg_action = action[start:end]
            seg_images = {k: v[start:end] for k, v in images.items()}
            
            output_path = os.path.join(output_dirs['cooperative'], f'episode_{episode_idx}_seg_{segment_count}')
            os.makedirs(output_dirs['cooperative'], exist_ok=True)
            save_hdf5(output_path, seg_qpos, seg_qvel, seg_action, seg_images, camera_names)
            segment_count += 1
    
    return segment_count


def main(args):
    dataset_dir = args['dataset_dir']
    num_episodes = args['num_episodes']
    
    # Output directories
    output_dirs = {
        'left_independent': dataset_dir + '_left_independent',
        'right_independent': dataset_dir + '_right_independent',
        'cooperative': dataset_dir + '_cooperative'
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
