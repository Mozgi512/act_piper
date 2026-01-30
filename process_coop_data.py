import os
import h5py
import numpy as np
import argparse
import time
from utils import apply_rgb_mask_to_strip, apply_rgb_mask_to_right_strip

def save_segment(episode_idx, qpos, qvel, action, images, camera_names, output_dir, mode, img_w):
    """
    mode: 'raw' or 'split'
    img_w: current image width (e.g. 640 or 320)
    """
    
    # Define constants based on resolution
    # Standard: 640 width -> split at 320, mask 40
    # Half: 320 width -> split at 160, mask 20
    scale = img_w / 640.0
    SPLIT_COL = int(320 * scale)
    MASK_WIDTH = int(40 * scale)
    
    if mode == 'c' or mode == 'raw':
        os.makedirs(output_dir, exist_ok=True)
        save_hdf5(os.path.join(output_dir, f'episode_{episode_idx}'), qpos, qvel, action, images, camera_names)
        
    elif mode == 'i' or mode == 'split':
        # Create Left and Right directories
        left_dir = output_dir + '_left'
        right_dir = output_dir + '_right'
        os.makedirs(left_dir, exist_ok=True)
        os.makedirs(right_dir, exist_ok=True)
        
        # --- Left ---
        # Indices 0-6 (7 dim) for Piper/Aloha standard
        p_left_qpos = qpos[:, :7]
        p_left_qvel = qvel[:, :7]
        p_left_action = action[:, :7]
        
        p_left_images = {}
        for cam_name, img_seq in images.items():
            # Crop Left Half: [:, :SPLIT_COL, :]
            cropped = img_seq[:, :, :SPLIT_COL, :]
            
            # Apply Mask to Right Edge
            masked_seq = []
            for i in range(cropped.shape[0]):
                frame = cropped[i].copy()
                # apply_rgb_mask_to_right_strip expects fixed 40 for 640. 
                # We need to adapt it or passed strip_width logic.
                # The utils function signature: apply_rgb_mask_to_right_strip(image, strip_width=40)
                frame = apply_rgb_mask_to_right_strip(frame, strip_width=MASK_WIDTH)
                masked_seq.append(frame)
            p_left_images[cam_name] = np.array(masked_seq, dtype=np.uint8)

        save_hdf5(os.path.join(left_dir, f'episode_{episode_idx}'), 
                  p_left_qpos, p_left_qvel, p_left_action, p_left_images, camera_names)

        # --- Right ---
        # Indices 7-13
        p_right_qpos = qpos[:, 7:14]
        p_right_qvel = qvel[:, 7:14]
        p_right_action = action[:, 7:14]

        p_right_images = {}
        # Right crop logic
        # Standard: 320-40=280 to 640-40=600. 
        # Logic: We want the "Main" part of the right image, centered?
        # In ALOHA/Piper, the camera is single, shared.
        # Right arm view is the right half.
        # Original code:
        # OFFSET = 40
        # START_COL = 320 - OFFSET # 280
        # END_COL = 640 - OFFSET   # 600
        # This shifts the window to the left? or just crops excluding black borders?
        # Actually it seems to try to center the view for the arm.
        
        OFFSET = MASK_WIDTH
        START_COL = SPLIT_COL - OFFSET
        END_COL = img_w - OFFSET
        
        for cam_name, img_seq in images.items():
            cropped = img_seq[:, :, START_COL:END_COL, :]
            
            masked_seq = []
            for i in range(cropped.shape[0]):
                frame = cropped[i].copy()
                if OFFSET > 0:
                    frame = apply_rgb_mask_to_strip(frame, strip_width=OFFSET)
                masked_seq.append(frame)
            p_right_images[cam_name] = np.array(masked_seq, dtype=np.uint8)

        save_hdf5(os.path.join(right_dir, f'episode_{episode_idx}'), 
                  p_right_qpos, p_right_qvel, p_right_action, p_right_images, camera_names)


def process_episode(episode_idx, dataset_dir, steps, types, camera_names, output_dirs):
    dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
    
    if not os.path.exists(dataset_path):
        print(f"Episode {episode_idx} not found at {dataset_path}")
        return False

    with h5py.File(dataset_path, 'r') as root:
        qpos = root['/observations/qpos'][()]
        qvel = root['/observations/qvel'][()]
        action = root['/action'][()]
        
        images = {}
        for cam_name in camera_names:
            images[cam_name] = root[f'/observations/images/{cam_name}'][()]

    # Detect Resolution
    sample_img = list(images.values())[0]
    img_h, img_w, _ = sample_img.shape[1:] # (T, H, W, C)

    total_len = qpos.shape[0]
    
    # Process each phase
    # steps: [split1, split2]
    # phases: [0->s1, s1->s2, s2->end]
    
    boundaries = [0] + steps + [total_len]
    # boundaries e.g. [0, 100, 300, 1000]
    
    for i, mode in enumerate(types):
        if mode == 'none':
            continue
            
        start = boundaries[i]
        end = boundaries[i+1]
        
        if start >= total_len:
            break # No data for this phase
            
        if end > total_len:
            end = total_len
        
        if start >= end:
            continue

        # Extract data
        p_qpos = qpos[start:end]
        p_qvel = qvel[start:end]
        p_action = action[start:end]
        p_images = {k: v[start:end] for k, v in images.items()}
        
        save_segment(episode_idx, p_qpos, p_qvel, p_action, p_images, camera_names, output_dirs[i], mode, img_w)

    return True

def save_hdf5(dataset_path, qpos, qvel, action, images, camera_names):
    max_timesteps = qpos.shape[0]
    if len(images) > 0:
        sample_img = list(images.values())[0]
        img_h, img_w = sample_img.shape[1], sample_img.shape[2]
    else:
        img_h, img_w = 480, 640 # Default fallback
        
    with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
        root.attrs['sim'] = True
        obs = root.create_group('observations')
        image_grp = obs.create_group('images')
        
        for cam_name in camera_names:
            _ = image_grp.create_dataset(cam_name, (max_timesteps, img_h, img_w, 3), dtype='uint8',
                                     chunks=(1, img_h, img_w, 3), )
        
        obs.create_dataset('qpos', data=qpos)
        obs.create_dataset('qvel', data=qvel)
        root.create_dataset('action', data=action)
        
        for cam_name, img_data in images.items():
            root[f'/observations/images/{cam_name}'][...] = img_data


def main(args):
    dataset_dir = args['dataset_dir']
    num_episodes = args['num_episodes']
    
    # Parse Split Steps
    # Using simple args for updating existing usage
    # old args: split_step (p1 end), phase2_len
    # new generic args mechanism
    splits = []
    if args['split1'] is not None:
        splits.append(args['split1'])
    if args['split2'] is not None:
        splits.append(args['split2'])
        
    types = [args['type1'], args['type2'], args['type3']]
    
    # Defaults mapping for backward compatibility if needed, but here we define new args
    
    # Output Directories
    output_dirs = []
    phase_names = ['phase1', 'phase2', 'phase3']
    for i, name in enumerate(phase_names):
        output_dirs.append(dataset_dir + f'_{name}')
        # os.makedirs(output_dirs[-1], exist_ok=True) # Logic moved to save_segment to avoid empty dirs
    
    print(f"Processing data from {dataset_dir}")
    print(f"Splits: {splits}")
    print(f"Types: {types}")
    
    # Inspect first episode
    first_ep_path = os.path.join(dataset_dir, 'episode_0.hdf5')
    if not os.path.exists(first_ep_path):
        print(f"Error: Could not find {first_ep_path}")
        return

    with h5py.File(first_ep_path, 'r') as f:
        camera_names = list(f['/observations/images'].keys())
    
    count = 0
    t0 = time.time()
    for i in range(num_episodes):
        if process_episode(i, dataset_dir, splits, types, camera_names, output_dirs):
            count += 1
        if (i+1) % 10 == 0:
            print(f"Processed {i+1}/{num_episodes} episodes...")

    print(f"Finished processing {count} episodes in {time.time() - t0:.1f} seconds.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', action='store', type=str, required=True)
    parser.add_argument('--num_episodes', action='store', type=int, required=True)
    
    # Split points
    parser.add_argument('--split1', action='store', type=int, help='End of Phase 1')
    parser.add_argument('--split2', action='store', type=int, help='End of Phase 2 (Start of Phase 3)')
    
    # Types: 'c' (Cooperation/Raw), 'i' (Independent/Split), 'none'
    parser.add_argument('--type1', action='store', type=str, default='none', help='Type for Phase 1 (c, i, or none)')
    parser.add_argument('--type2', action='store', type=str, default='none', help='Type for Phase 2 (c, i, or none)')
    parser.add_argument('--type3', action='store', type=str, default='none', help='Type for Phase 3 (c, i, or none)')

    main(vars(parser.parse_args()))
