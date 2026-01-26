import os
import h5py
import numpy as np
import argparse
import time
from utils import apply_rgb_mask_to_strip
import cv2

def process_episode(episode_idx, dataset_dir, phase1_dir, phase2_left_dir, phase2_right_dir, camera_names, split_step, phase2_len):
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

    # --- Phase 1: t=0 to split_step ---
    PHASE1_END = split_step
    
    # Slicing
    p1_qpos = qpos[:PHASE1_END]
    p1_qvel = qvel[:PHASE1_END]
    p1_action = action[:PHASE1_END]
    p1_images = {k: v[:PHASE1_END] for k, v in images.items()}
    
    # Save Phase 1
    save_hdf5(os.path.join(phase1_dir, f'episode_{episode_idx}'), p1_qpos, p1_qvel, p1_action, p1_images, camera_names)

    # --- Phase 2: t=split_step to split_step + phase2_len ---
    PHASE2_START = split_step
    PHASE2_END = split_step + phase2_len
    
    # Verify length
    total_len = qpos.shape[0]
    if total_len < PHASE2_END:
        print(f"Warning: Episode {episode_idx} length {total_len} is shorter than required {PHASE2_END}. Clipping end.")
        PHASE2_END = total_len

    if PHASE2_START >= total_len:
         print(f"Warning: Episode {episode_idx} too short for Phase 2 ({total_len} < {PHASE2_START}). Skipping Phase 2.")
         return True


    # Raw Phase 2 Data
    p2_raw_qpos = qpos[PHASE2_START:PHASE2_END]
    p2_raw_qvel = qvel[PHASE2_START:PHASE2_END]
    p2_raw_action = action[PHASE2_START:PHASE2_END]
    p2_raw_images = {k: v[PHASE2_START:PHASE2_END] for k, v in images.items()}

    # --- Phase 2 Left ---
    # Indices 0-6 (7 dim)
    p2_left_qpos = p2_raw_qpos[:, :7]
    p2_left_qvel = p2_raw_qvel[:, :7]
    p2_left_action = p2_raw_action[:, :7]
    
    from utils import apply_rgb_mask_to_right_strip # Import here or top level
    
    p2_left_images = {}
    for cam_name, img_seq in p2_raw_images.items():
        # Crop Left Half: [:, :320, :]
        cropped = img_seq[:, :, :320, :]
        
        # Apply Mask to Right Edge (40px)
        masked_seq = []
        for i in range(cropped.shape[0]):
            frame = cropped[i].copy()
            frame = apply_rgb_mask_to_right_strip(frame, strip_width=40)
            masked_seq.append(frame)
        
        p2_left_images[cam_name] = np.array(masked_seq, dtype=np.uint8)

    save_hdf5(os.path.join(phase2_left_dir, f'episode_{episode_idx}'), 
              p2_left_qpos, p2_left_qvel, p2_left_action, p2_left_images, camera_names)

    # --- Phase 2 Right ---
    # Indices 7-13 (7 dim)
    p2_right_qpos = p2_raw_qpos[:, 7:14]
    p2_right_qvel = p2_raw_qvel[:, 7:14]
    p2_right_action = p2_raw_action[:, 7:14]

    p2_right_images = {}
    OFFSET = 40
    START_COL = 320 - OFFSET # 280
    END_COL = 640 - OFFSET   # 600
    
    for cam_name, img_seq in p2_raw_images.items():
        # Crop Right with Shift: [:, 280:600, :]
        cropped = img_seq[:, :, START_COL:END_COL, :]
        
        # Apply Mask
        # We need to apply mask to each frame. Doing it in a loop or vectorizing?
        # utils.apply_rgb_mask_to_strip takes (H, W, 3).
        # We have (T, H, W, 3).
        masked_seq = []
        for i in range(cropped.shape[0]):
            frame = cropped[i].copy() # Ensure writable
            if OFFSET > 0:
                frame = apply_rgb_mask_to_strip(frame, strip_width=OFFSET)
            masked_seq.append(frame)
        p2_right_images[cam_name] = np.array(masked_seq, dtype=np.uint8)

    save_hdf5(os.path.join(phase2_right_dir, f'episode_{episode_idx}'), 
              p2_right_qpos, p2_right_qvel, p2_right_action, p2_right_images, camera_names)

    return True

def save_hdf5(dataset_path, qpos, qvel, action, images, camera_names):
    max_timesteps = qpos.shape[0]
    # Check image dims
    if len(images) > 0:
        sample_img = list(images.values())[0]
        img_h, img_w = sample_img.shape[1], sample_img.shape[2]
    else:
        # Fallback if no images (should not happen usually)
        img_h, img_w = 480, 640
        
    action_dim = action.shape[1]

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
    split_step = args['split_step']
    phase2_len = args['phase2_len']
    
    # Output Directories
    phase1_dir = dataset_dir + '_phase1'
    phase2_left_dir = dataset_dir + '_phase2_left'
    phase2_right_dir = dataset_dir + '_phase2_right'
    
    os.makedirs(phase1_dir, exist_ok=True)
    os.makedirs(phase2_left_dir, exist_ok=True)
    os.makedirs(phase2_right_dir, exist_ok=True)
    
    print(f"Processing data from {dataset_dir}")
    print(f"Split Step: {split_step}")
    print(f"Phase 2 Length: {phase2_len} (End: {split_step + phase2_len})")
    print(f"Output Phase 1: {phase1_dir}")
    print(f"Output Phase 2 Left: {phase2_left_dir}")
    print(f"Output Phase 2 Right: {phase2_right_dir}")

    # Inspect first episode to get camera names
    first_ep_path = os.path.join(dataset_dir, 'episode_0.hdf5')
    if not os.path.exists(first_ep_path):
        print(f"Error: Could not find {first_ep_path}")
        return

    with h5py.File(first_ep_path, 'r') as f:
        camera_names = list(f['/observations/images'].keys())
    print(f"Camera names found: {camera_names}")

    count = 0
    t0 = time.time()
    for i in range(num_episodes):
        if process_episode(i, dataset_dir, phase1_dir, phase2_left_dir, phase2_right_dir, camera_names, split_step, phase2_len):
            count += 1
        if (i+1) % 10 == 0:
            print(f"Processed {i+1}/{num_episodes} episodes...")

    print(f"Finished processing {count} episodes in {time.time() - t0:.1f} seconds.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', action='store', type=str, help='Source dataset directory', required=True)
    parser.add_argument('--num_episodes', action='store', type=int, help='Number of episodes', required=True)
    parser.add_argument('--split_step', action='store', type=int, default=280, help='Timestep to split Phase 1 and Phase 2')
    parser.add_argument('--phase2_len', action='store', type=int, default=300, help='Length of Phase 2')
    
    main(vars(parser.parse_args()))
