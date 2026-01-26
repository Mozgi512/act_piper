
import time
import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
import h5py

from piper_constants import PUPPET_GRIPPER_POSITION_NORMALIZE_FN, SIM_TASK_CONFIGS
from piper_ee_sim_env import make_ee_sim_env
from piper_sim_env import make_sim_env, REDBOX_POSE, GREENBOX_POSE, BLUEBOX_POSE
from scripted_policy import VariableCoopPolicy

import IPython

def main():
    # Configuration
    parser = argparse.ArgumentParser()
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--target_order', action='store', type=str, nargs='+', help='Specific permutations to generate (e.g. RGB BRG)')
    args = parser.parse_args()
    
    base_dataset_dir = 'data/variable_coop_dataset'
    task_name = 'sim_coop_scripted' # Use coop config for camera/length
    num_episodes_per_config = 20
    onscreen_render = args.onscreen_render
    
    # 6 Permutations of Cube IDs [7=R, 8=G, 9=B]
    # We map these IDs to 3 spatial slots/positions on the belt.
    # Slot 1 (Downstream/Right), Slot 2 (Middle), Slot 3 (Upstream/Left)
    # If Belt moves Left->Right (+X?), then Rightmost X arrives first?
    # Let's assume Slots X coords: [0.25, 0.0, -0.25]
    
    permutations = {
        'RGB': [7, 8, 9], # R at 0.25, G at 0.0, B at -0.25
        'RBG': [7, 9, 8],
        'GRB': [8, 7, 9],
        'GBR': [8, 9, 7],
        'BRG': [9, 7, 8],
        'BGR': [9, 8, 7]
    }

    # Filter permutations if target_order is specified
    if args.target_order:
        new_permutations = {}
        for order in args.target_order:
            if order in permutations:
                new_permutations[order] = permutations[order]
            else:
                print(f"Warning: Unknown permutation '{order}', skipping.")
        permutations = new_permutations
        if not permutations:
            print("No valid permutations selected. Exiting.")
            return
    
    # X-coordinates for the 3 slots
    # Standard spacing is 0.22. let's use [0.22, 0.0, -0.22] + slight random noise?
    # To be safe, let's stick to the generated logic but override the IDs.
    
    slots_x = [0.20, -0.02, -0.24] 
    
    if not os.path.exists(base_dataset_dir):
        os.makedirs(base_dataset_dir)
        
    camera_names = ['top'] # Hardcoded standard
    
    for perm_name, ordered_ids in permutations.items():
        print(f"Generating data for permutation: {perm_name} ...")
        dataset_dir = os.path.join(base_dataset_dir, perm_name)
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
            
        success_count = 0
        
        for episode_idx in range(num_episodes_per_config):
            print(f"  Episode {episode_idx}")
            
            # --- Rollout (EE Space) ---
            env = make_ee_sim_env('sim_variable_coop', camera_names=camera_names) # Base env
            ts = env.reset()
            
            # OVERRIDE CUBE POSITIONS
            physics = env.physics
            # ordered_ids = [First, Second, Third]
            # Assign ordered_ids[0] to slot 0 (Rightmost/First)
            
            for i, cube_id in enumerate(ordered_ids):
                # Find joint address
                joint_name = f'cube_{cube_id}_joint'
                joint_id = physics.model.name2id(joint_name, 'joint')
                addr = physics.model.jnt_qposadr[joint_id]
                
                # Set Position
                # X: Slot value + small noise
                x_val = slots_x[i] + np.random.uniform(-0.01, 0.01)
                # Y: 0.35 + small noise
                y_val = 0.35 + np.random.uniform(-0.02, 0.02)
                # Z: 0.025
                z_val = 0.025
                
                new_pose = np.array([x_val, y_val, z_val, 1, 0, 0, 0])
                physics.data.qpos[addr:addr+7] = new_pose

            # Fix for duplicates: Position Queue Cubes (0-6) relative to Left-Most Active Cube
            
            # 1. Find the X positions of 7, 8, 9
            # We already set them in the loop above.
            # To be robust, let's just query them from physics.
            x_positions = []
            for target_id in [7, 8, 9]:
                j_name = f'cube_{target_id}_joint'
                j_id = physics.model.name2id(j_name, 'joint')
                addr = physics.model.jnt_qposadr[j_id]
                x_positions.append(physics.data.qpos[addr]) # X is first element
            
            ref_x = min(x_positions)
            
            queue_spacing = 0.22
            
            for i in range(7): # 0 to 6
                # Position relative to Left-Most Active Cube
                # step = 7 - i # 7 is the "head" index of queue (virtual 7th position)
                # Here, we treat 'ref_x' as the head position (virtual 7/8/9 position, whichever is left-most).
                
                step = 7 - i 
                
                cube_x = ref_x - step * queue_spacing + np.random.uniform(-0.01, 0.01)
                cube_y = 0.35 + np.random.uniform(-0.02, 0.02)
                cube_z = 0.025
                
                joint_name = f'cube_{i}_joint'
                joint_id = physics.model.name2id(joint_name, 'joint')
                addr = physics.model.jnt_qposadr[joint_id]
                
                new_pose = np.array([cube_x, cube_y, cube_z, 1, 0, 0, 0])
                physics.data.qpos[addr:addr+7] = new_pose
            
            # Manual Observation Update
            ts_obs = env.task.get_observation(physics)
            # Re-package into TimeStep-like object if needed, or update existing `ts`
            import dm_env
            ts = dm_env.TimeStep(
                step_type=ts.step_type,
                reward=ts.reward,
                discount=ts.discount,
                observation=ts_obs
            )
            
            episode = [ts]
            
            # Initialize Policy
            policy = VariableCoopPolicy(inject_noise=False)
            
            # Recording loop
            episode_len = 1000 # Sufficient for coop
            all_actions = []
            
            for step in range(episode_len):
                try:
                    action = policy(ts)
                except Exception as e:
                    print(f"Policy Error: {e}")
                    break
                    
                all_actions.append(action)
                ts = env.step(action)
                episode.append(ts)
                
                if policy.success_t is not None and step >= policy.success_t:
                    break

                if onscreen_render:
                    if step == 0:
                        plt.ion()
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
                        img_top = ax1.imshow(ts.observation['images']['top'])
                        ax1.set_title("Top Realtime")
                        img_angle = ax2.imshow(ts.observation['images']['angle'])
                        ax2.set_title("Angle (Not Recorded)")
                        plt.pause(0.1)
                    else:
                        if step % 5 == 0:
                            img_top.set_data(ts.observation['images']['top'])
                            img_angle.set_data(ts.observation['images']['angle'])
                            plt.pause(0.001)
                
            # Check Success
            # Max reward for coop is 4?
            episode_max_reward = np.max([t.reward for t in episode[1:]])
            if episode_max_reward == 4: # Assuming 4 is success for coop
                # print("Success")
                success_count += 1
            
            # Extract Joint Traj
            joint_traj = [t.observation['qpos'] for t in episode]
            gripper_ctrl_traj = [t.observation['gripper_ctrl'] for t in episode]
            
            # Replace gripper (EE control -> Joint control)
            for j, ctrl in zip(joint_traj, gripper_ctrl_traj):
                left_ctrl = PUPPET_GRIPPER_POSITION_NORMALIZE_FN(ctrl[0])
                right_ctrl = PUPPET_GRIPPER_POSITION_NORMALIZE_FN(ctrl[1])
                j[6] = left_ctrl
                j[6+7] = right_ctrl
                
            subtask_info = episode[0].observation['env_state'].copy()
            
            # Clean up
            del env, episode, policy
            
            # --- Replay (Sim Env) ---
            # To save consistent data with camera renders
            # We must set the SAME initial positions.
            print("  Replaying...")
            env = make_sim_env('sim_variable_coop', camera_names=camera_names)
            
            # Update Global Poses for Replay Init
            # box_pose is index 49:56? Check piper_ee_sim_env get_env_state offset
            # But here we can just use the values we set, OR copy from subtask_info
            # In 'sim_coop', subtask_info has 70 dims (10 cubes).
            # R=7 (index 7*7=49?), G=8 (56), B=9 (63).
            # We rely on `record_sim_episodes.py` logic which handles this.
            # But `record_sim_episodes.py` uses `REDBOX_POSE`, etc.
            # We need to manually inject our permutations into REDBOX_POSE etc. before reset.
            
            # Extract from subtask_info
            # Assuming env_state starts with 10 cubes (0-9).
            # Cube 7 is 7*7 = 49.
            # Cube 8 is 56.
            # Cube 9 is 63.
            REDBOX_POSE[0] = subtask_info[49:56]
            GREENBOX_POSE[0] = subtask_info[56:63]
            BLUEBOX_POSE[0] = subtask_info[63:70]
            
            ts = env.reset()
            episode_replay = [ts]
            
            for t in range(len(joint_traj)):
                action = joint_traj[t].copy() # Joint position action
                ts = env.step(action)
                episode_replay.append(ts)

                if onscreen_render:
                    if t == 0:
                        plt.ion()
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
                        img_top = ax1.imshow(ts.observation['images']['top'])
                        ax1.set_title("Top Replay")
                        img_angle = ax2.imshow(ts.observation['images']['angle'])
                        ax2.set_title("Angle Replay")
                        plt.pause(0.1)
                    else:
                        if t % 5 == 0:
                            img_top.set_data(ts.observation['images']['top'])
                            img_angle.set_data(ts.observation['images']['angle'])
                            plt.pause(0.001)
                
            # Save
            data_dict = {
                '/observations/qpos': [],
                '/observations/qvel': [],
            }
            for cam_name in camera_names:
                data_dict[f'/observations/images/{cam_name}'] = []
                
            max_timesteps = len(joint_traj) - 1 # Truncate last like original script
            
            for i in range(max_timesteps):
                data_dict['/observations/qpos'].append(episode_replay[i].observation['qpos'])
                data_dict['/observations/qvel'].append(episode_replay[i].observation['qvel'])
                data_dict['/action'].append(joint_traj[i])
                for cam_name in camera_names:
                    data_dict[f'/observations/images/{cam_name}'].append(episode_replay[i].observation['images'][cam_name])
            
            # HDF5 Write
            dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}')
            with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
                root.attrs['sim'] = True
                obs = root.create_group('observations')
                image = obs.create_group('images')
                for cam_name in camera_names:
                    _ = image.create_dataset(cam_name, (max_timesteps, 480, 640, 3), dtype='uint8', chunks=(1, 480, 640, 3))
                
                qpos = obs.create_dataset('qpos', (max_timesteps, 14))
                qvel = obs.create_dataset('qvel', (max_timesteps, 14))
                action = root.create_dataset('action', (max_timesteps, 14))
                
                for name, array in data_dict.items():
                    root[name][...] = array
            
            del env, episode_replay
            
        print(f"Permuation {perm_name} Done. Success count in EE rollout: {success_count}/{num_episodes_per_config}")

if __name__ == "__main__":
    main()
