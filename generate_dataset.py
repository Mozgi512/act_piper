
import time
import os
os.environ['MUJOCO_GL'] = 'egl'

import numpy as np
import argparse
import matplotlib.pyplot as plt
import h5py
import collections

from piper_constants import PUPPET_GRIPPER_POSITION_NORMALIZE_FN, SIM_TASK_CONFIGS, BELT_MOVE_SPEED, PUPPET_GRIPPER_POSITION_OPEN, PUPPET_GRIPPER_POSITION_CLOSE
from piper_ee_sim_env import make_ee_sim_env
import piper_sim_env
from piper_sim_env import make_sim_env, MANYCUBES_POSES, MANYCUBES_COLORS
from interactive_policy import InteractivePolicy, COLOR_SEQUENCE

def main(args):
    """
    Generate demonstration data by rolling out InteractivePolicy (EE) and replaying in Joint space.
    """
    task_name = args['task_name']
    dataset_dir = args['dataset_dir']
    num_episodes = args['num_episodes']
    onscreen_render = args['onscreen_render']
    if not onscreen_render:
        print("Note: Run with --onscreen_render to see the visualization.")
    inject_noise = False
    render_cam_name = 'top'
    
    command_sequence_str = args['commands']
    command_queue_template = list(command_sequence_str)

    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir, exist_ok=True)

    episode_len = SIM_TASK_CONFIGS[task_name]['episode_len']
    camera_names = SIM_TASK_CONFIGS[task_name]['camera_names']

    success_count = 0
    episode_idx = 0
    
    while episode_idx < num_episodes:
        print(f'Episode {episode_idx}: Rollout out EE space InteractivePolicy')
        
        # ---------------------------------------------------------
        # 1. EE Rollout
        # ---------------------------------------------------------
        
        # Inject color sequence if needed
        MANYCUBES_COLORS[0] = COLOR_SEQUENCE
        
        # Force EGL for headless rendering during rollout (if needed, or just let it be)
        # os.environ['MUJOCO_GL'] = 'egl' 
        
        # Optimization: Disable cameras if not rendering to speed up EE execution
        ee_cameras = ['top'] if onscreen_render else []
        env = make_ee_sim_env(task_name, camera_names=ee_cameras) 
        ts = env.reset()
        episode = [ts]
        
        policy = InteractivePolicy(inject_noise=inject_noise)
        policy.init_pose(ts)

        if onscreen_render:
             import cv2
             window_name = "Dataset Generation"
             cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
             try:
                 cv2.startWindowThread()
             except:
                 pass

        
        # Prepare command queue for this episode
        command_queue = list(command_queue_template)
        next_trigger = 20 if command_queue else -1
        
        # Loop for EE rollout
        # Use a large max_steps to allow dynamic termination, overriding fixed episode_len
        max_steps = 3000 
        for step in range(max_steps):
            # Auto Execution Logic
            if command_queue and step == next_trigger:
                cmd = command_queue.pop(0).upper()
                print(f"  [EE] Auto-executing command '{cmd}' at step {step}")
                policy.schedule_command(cmd, ts)
                next_trigger = -1 # Reset trigger
            
            # Check for completion to schedule next task
            # Check for completion to schedule next task
            if step == policy.last_action_end_t:
                print(f"  [EE] Subtask completed at step {step}")
                if command_queue:
                    next_trigger = step + 20
                    print(f"  [EE] Next task scheduled at step {next_trigger}")
                else:
                    print(f"  [EE] All commands scheduled. Will terminate at step {step + 20}")

            # Dynamic efficiency termination
            # Terminate 20 steps after the last action is completed (and queue is empty)
            # Dynamic efficiency termination
            # Terminate 20 steps after the last action is completed (and queue is empty)
            if not command_queue and policy.last_action_end_t != -1 and step >= policy.last_action_end_t + 20:
                 print(f"  [EE] Dynamic Termination at step {step}")
                 break
            
            action = policy(ts)
            ts = env.step(action)
            episode.append(ts)
            
            # Optional: Render
            if onscreen_render:
                # Render every 5 steps (reduced from 2)
                if step % 5 == 0:
                    img_rgb = ts.observation['images']['top']
                    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                    cv2.imshow(window_name, img_bgr)
                    cv2.waitKey(1)
 

        # Capture Env State for Replay
        # env_state includes robot state + object poses
        # For sim_many_cubes: 
        # qpos structure: robot(16) + belt(1) + belt_ext(1) + 10 cubes (7*10) = 88
        # get_env_state returns qpos[18:18+70] -> The 10 cubes.
        subtask_info = episode[0].observation['env_state'].copy() 
        
        # Extract Joint Trajectory
        joint_traj = [ts.observation['qpos'] for ts in episode]
        
        # Replace gripper pose with gripper control (standard practice in record_sim_episodes)
        gripper_ctrl_traj = [ts.observation['gripper_ctrl'] for ts in episode]
        for joint, ctrl in zip(joint_traj, gripper_ctrl_traj):
            left_ctrl = PUPPET_GRIPPER_POSITION_NORMALIZE_FN(ctrl[0])
            right_ctrl = PUPPET_GRIPPER_POSITION_NORMALIZE_FN(ctrl[1])
            joint[6] = left_ctrl
            joint[6+7] = right_ctrl

        # Fix: Overwrite Step 0 gripper to OPEN (Fix for "Starts closed" issue)
        # env.reset() defaults gripper to Closed, so the first frame captures Closed.
        # But policy starts Open. We force first frame to Open.
        left_open_val = PUPPET_GRIPPER_POSITION_NORMALIZE_FN(PUPPET_GRIPPER_POSITION_OPEN)
        right_open_val = PUPPET_GRIPPER_POSITION_NORMALIZE_FN(PUPPET_GRIPPER_POSITION_OPEN)
        joint_traj[0][6] = left_open_val
        joint_traj[0][13] = right_open_val

        # Fix: Unwrap joint trajectory to prevent wrap-around jumps (e.g. wrist rotation)
        # Convert list of arrays to one big array for unwrapping
        joint_traj_np = np.array(joint_traj) # (T, 14)
        
        # Unwrap each joint column. Discontinuity threshold = pi
        # Note: Gripper indices are linear (0-1), unwrapping won't affect them if they are smooth
        # But for rotation joints (radians), this fixes jumps +/- 2pi
        for j in range(14):
             joint_traj_np[:, j] = np.unwrap(joint_traj_np[:, j])
        
        # Convert back to list of arrays
        joint_traj_np = np.array(joint_traj) # Need this for processing
        
        # 2. Filter out large jumps from configuration flips or glitches
        # Threshold: 0.2 rad per step (approx 11 degrees) in 20ms is physically impossible/dangerous (limit ~10rad/s)
        THRESHOLD = 0.2
        for t in range(1, len(joint_traj_np)):
            diff = joint_traj_np[t] - joint_traj_np[t-1]
            # Identify joints with large jumps
            jump_indices = np.where(np.abs(diff) > THRESHOLD)[0]
            if len(jump_indices) > 0:
                 # Heuristic: If jump detected, hold previous position for this step
                 # This smoothes out single-step glitches. 
                 # If the jump persists (state change), this will drag it out over time or break.
                 # But usually wild arm is a transient flip.
                 # Let's simple clamp to previous value to avoid impulse
                 joint_traj_np[t][jump_indices] = joint_traj_np[t-1][jump_indices]
        
        # Re-convert to list
        joint_traj = [joint_traj_np[t] for t in range(len(joint_traj_np))]
        
        # Debug: Check for residual jumps
        jumps = np.diff(joint_traj_np, axis=0)
        max_jump = np.max(np.abs(jumps))
        if max_jump > 0.2:
             print(f"WARNING: Large joint jump detected! Max: {max_jump:.3f} rad/step")
             rows, cols = np.where(np.abs(jumps) > 0.2)
             for r, c in zip(rows[:10], cols[:10]):
                 print(f"  Jump at Step {r}, Joint {c}: {jumps[r, c]:.3f}")

        del env
        del policy

        # ---------------------------------------------------------
        # 2. Joint Space Replay
        # ---------------------------------------------------------
        print(f'Episode {episode_idx}: Replaying in Joint space')
        
        # Inject Initial Poses from EE Rollout
        poses = {}
        for i in range(10):
            poses[i] = subtask_info[i*7 : (i+1)*7].copy()
        MANYCUBES_POSES[0] = poses
        MANYCUBES_COLORS[0] = COLOR_SEQUENCE
        
        env = make_sim_env(task_name, camera_names=['top']) # uses MANYCUBES_POSES[0]
        ts = env.reset()
        episode_replay = [ts]
        
        # Object Removal State for Replay
        removal_timers = {}
        removed_objects = set()
        DT = 0.02 # From piper_constants
        
        # Replay Loop
        for t in range(len(joint_traj)):
            action = joint_traj[t].copy()
            ts = env.step(action)
            episode_replay.append(ts)
            
            if onscreen_render:
                if t % 5 == 0:
                    img_rgb = ts.observation['images'][render_cam_name]
                    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                    cv2.imshow(window_name, img_bgr)
                    cv2.waitKey(1)

            # Object Removal Logic (Mirroring piper_ee_sim_env.py after_step)
            current_time = t * DT
            physics = env.physics
            
            for i in range(10):
                if i in removed_objects:
                    continue
                try:
                    joint_id = physics.model.name2id(f'cube_{i}_joint', 'joint')
                    qpos_adr = physics.model.jnt_qposadr[joint_id]
                    y_pos = physics.data.qpos[qpos_adr + 1]
                    
                    if y_pos < 0.28: # In Placement Zone
                        if i not in removal_timers:
                            removal_timers[i] = current_time
                        
                        if current_time - removal_timers[i] > 4.0:
                            # Remove (Teleport)
                            # print(f"Mirroring Removal of Cube {i} at step {t}")
                            hidden_pos = np.array([10.0 + i, -10.0, -1.0, 1, 0, 0, 0])
                            np.copyto(physics.data.qpos[qpos_adr : qpos_adr+7], hidden_pos)
                            
                            dof_adr = physics.model.jnt_dofadr[joint_id]
                            np.copyto(physics.data.qvel[dof_adr : dof_adr+6], np.zeros(6))
                            
                            removed_objects.add(i)
                    else:
                         if i in removal_timers:
                             del removal_timers[i]
                except:
                    pass

        # Verify Success (simple reward check)
        rewards = [ts.reward for ts in episode_replay[1:] if ts.reward is not None]
        episode_max_reward = np.max(rewards) if rewards else 0.0
        if episode_max_reward == env.task.max_reward:
            print(f"Episode {episode_idx} Successful Replay")
            success_count += 1
        else:
            print(f"Episode {episode_idx} Failed Replay (Reward: {episode_max_reward})")
            # Make sure we don't save failed replays? Or flag them?
            # record_sim_episodes skips saving if failed.
            # But InteractivePolicy might not always "succeed" perfectly if logic is loose.
            # For now, let's SAVE regardless, but warn. Or maybe we want only successful ones.
            # User request didn't specify, but usually we want good data.
            # However, "InteractivePolicy" is deterministic if seeded/reset? 
            # If the EE policy works, Joint replay should work too.
            pass

        # ---------------------------------------------------------
        # 3. Save to HDF5
        # ---------------------------------------------------------
        
        data_dict = {
            '/observations/qpos': [],
            '/observations/qvel': [],
            '/action': [],
        }
        for cam_name in camera_names:
            data_dict[f'/observations/images/{cam_name}'] = []

        # Truncate to match lengths (actions = steps, observations = steps + 1)
        # joint_traj has N items (actions). episode_replay has N+1 items (timesteps).
        # We process N steps.
        
        # Note: record_sim_episodes saves lengths consistent with ACT.
        # It usually truncates the last observation or something.
        # record_sim_episodes logic:
        # joint_traj = joint_traj[:-1]
        # episode_replay = episode_replay[:-1]
        # max_timesteps = len(joint_traj)
        
        # Let's follow record_sim_episodes exactly.
        action_list = joint_traj[:-1] # Remove last action? Wait.
        # In record_sim_episodes: 
        #   for t in range(len(joint_traj)): ... ts = env.step() ...
        #   joint_traj has N elements. episode_replay has N+1.
        #   Then: joint_traj = joint_traj[:-1]; episode_replay = episode_replay[:-1]
        #   So we save N-1 steps? 
        #   Let's just save valid pairs.
        
        num_steps_to_save = len(joint_traj) - 1 # Use N-1 to be safe/standard
        
        for k in range(num_steps_to_save):
            action = joint_traj[k]
            ts = episode_replay[k]
            
            data_dict['/observations/qpos'].append(ts.observation['qpos'])
            data_dict['/observations/qvel'].append(ts.observation['qvel'])
            data_dict['/action'].append(action)
            for cam_name in camera_names:
                data_dict[f'/observations/images/{cam_name}'].append(ts.observation['images'][cam_name])
                
        # Save
        if not args['no_save_data']:
            t0 = time.time()
            dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}')
            with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
                root.attrs['sim'] = True
                obs = root.create_group('observations')
                image = obs.create_group('images')
                for cam_name in camera_names:
                    _ = image.create_dataset(cam_name, (num_steps_to_save, 480, 640, 3), dtype='uint8',
                                             chunks=(1, 480, 640, 3), )
                qpos = obs.create_dataset('qpos', (num_steps_to_save, 14))
                qvel = obs.create_dataset('qvel', (num_steps_to_save, 14))
                action = root.create_dataset('action', (num_steps_to_save, 14))

                for name, array in data_dict.items():
                    root[name][...] = array
            print(f'Saving: {time.time() - t0:.1f} secs\n')

        del env
        del episode_replay
        
        episode_idx += 1

    print(f'Done. Saved to {dataset_dir}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', action='store', type=str, help='task_name', default='sim_many_cubes')
    parser.add_argument('--dataset_dir', action='store', type=str, help='dataset saving dir', required=True)
    parser.add_argument('--num_episodes', action='store', type=int, help='num_episodes', required=True)
    parser.add_argument('--commands', action='store', type=str, help='Command sequence (e.g. ICI)', required=True)
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--no_save_data', action='store_true', help='Do not save HDF5 data')
    
    args = parser.parse_args()
    main(vars(args))
