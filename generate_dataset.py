
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
from piper_sim_env import make_sim_env, MANYCUBES_POSES, MANYCUBES_COLORS, MANYCUBES_TASK_COUNT, MANYCUBES_CONFIG
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
    if command_sequence_str:
        command_queue_template = list(command_sequence_str)
    else:
        command_queue_template = []
    
    # Parse color sequence if provided
    color_seq = None
    if args.get('color_sequence'):
        color_seq = list(args['color_sequence'].lower())
        if len(color_seq) != 10:
            raise ValueError(f"Color sequence must be exactly 10 characters (got {len(color_seq)})")
        for c in color_seq:
            if c not in ['r', 'g', 'b']:
                raise ValueError(f"Invalid color '{c}' in sequence. Use only 'r', 'g', 'b'")

    if not args.get('pretrain_mode') and not args.get('sequence_file') and not command_sequence_str:
        parser.error("--commands is required unless --pretrain_mode or --sequence_file is specified.")


    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir, exist_ok=True)

    episode_len = SIM_TASK_CONFIGS[task_name]['episode_len']
    camera_names = SIM_TASK_CONFIGS[task_name]['camera_names']

    success_count = 0
    episode_idx = 0
    total_retries = 0
    max_retries_per_episode = 5
    failed_sequences_count = 0
    
    # Sequence File Reset Logic
    if args.get('sequence_file'):
        import csv
        seq_file = args['sequence_file']
        print(f"Resetting marks in {seq_file}...")
        rows = []
        with open(seq_file, 'r', newline='') as f:
            reader = csv.reader(f)
            rows = list(reader)
            
        # Clear specific mark column (3rd) or trim?
        # User said "Delete marks". Trimming to 2 columns is safest/cleanest.
        cleaned_rows = []
        for r in rows:
            cleaned_rows.append(r[:2])
            
        with open(seq_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(cleaned_rows)
            
    while episode_idx < num_episodes:
        print(f"DEBUG: Start of loop. episode_idx = {episode_idx}")
        retry_count = 0
        episode_success = False
        
        while retry_count < max_retries_per_episode and not episode_success:
            if retry_count > 0:
                print(f'Episode {episode_idx}: Retry attempt {retry_count}/{max_retries_per_episode}')
            else:
                print(f'Episode {episode_idx}: Rollout out EE space InteractivePolicy')
        
            # ---------------------------------------------------------
            # 1. EE Rollout
            # ---------------------------------------------------------
        

            # Global Random X-Shift
            # Normal: 0.0 to 0.10
            # Pre-train: 1.0 +/- 0.10
            if args.get('pretrain_mode'):
                x_shift = 1.0 + np.random.uniform(-0.10, 0.10)
            else:
                x_shift = np.random.uniform(0.00, 0.10)
            MANYCUBES_CONFIG['x_shift'] = x_shift
            
            # Prediction Base: Match simulation environment (start_x = 0.0)
            start_x = 0.0
            print(f"  [Gen] Global X-Shift: {x_shift:.3f} (Start X effectively {start_x + x_shift:.3f})")

            # Sequence File Logic
            if args.get('sequence_file'):
                import csv
                seq_file = args['sequence_file']
                
                # 1. Read all sequences
                rows = []
                with open(seq_file, 'r', newline='') as f:
                    reader = csv.reader(f)
                    rows = list(reader)
                
                # 2. Filter for available sequences
                # Normal mode: not 'success' and not 'failed' and not 'test'
                # Test mode: only those without any status (len < 3)
                available_indices = []
                for i, r in enumerate(rows):
                    if args.get('test'):
                        if len(r) < 3:
                            available_indices.append(i)
                    else:
                        if len(r) < 3 or (r[2] != 'success' and r[2] != 'failed' and r[2] != 'test'):
                            available_indices.append(i)
                
                if not available_indices:
                    print(f"  [Gen] No available {'UNUSED' if args.get('test') else ''} sequences in {seq_file}. Terminating.")
                    break
                
                # 3. Pick random
                seq_idx = np.random.choice(available_indices)
                selected_row = rows[seq_idx]
                
                color_seq_str = selected_row[0]
                command_seq_str = selected_row[1]
                
                print(f"  [Gen] Selected Sequence {seq_idx}: {color_seq_str} -> Cmds: {command_seq_str}")
                
                color_seq = list(color_seq_str)
                command_queue_template = list(command_seq_str)
                
                # 4. Mark processing (optional, but good for safety)
                # We update to 'processing' to avoid other processes picking it? 
                # (Simple script, maybe not needed, but good practice).
                if len(selected_row) < 3:
                    selected_row.append('processing')
                else:
                    selected_row[2] = 'processing'
                
                rows[seq_idx] = selected_row
                
                # Rewrite file
                with open(seq_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerows(rows)
                
                # Inject
                MANYCUBES_COLORS[0] = color_seq
                MANYCUBES_TASK_COUNT[0] = len(command_queue_template)

            elif args.get('pretrain_mode'):
                # Calculate projected X positions for all 10 cubes
                # Logic matches piper_ee_sim_env.py: px = (start_x - i * spacing) + shift_val
                start_x = 0.0
                spacing = 0.15
                xs = [(start_x - i * spacing) + x_shift for i in range(10)]
            
                # Identify spatial candidates
                if args.get('pretrain_mode') == 'cooperative':
                    # Loosened ranges for cooperative mode
                    l_bound_u, l_bound_l = 0.10, -0.50
                    r_bound_l, r_bound_u = -0.30, 0.50
                else:
                    # Strict ranges for independent mode
                    l_bound_u, l_bound_l = -0.10, -0.40
                    r_bound_l, r_bound_u = 0.00, 0.30

                left_candidates = [i for i, x in enumerate(xs) if x <= l_bound_u and x > l_bound_l]
                right_candidates = [i for i, x in enumerate(xs) if x >= r_bound_l and x < r_bound_u]
            
                if not left_candidates or not right_candidates:
                    # Fallback if shift/spacing pushes everything too far
                    print(f"  [Gen] WARNING: Could not find candidates in both regions (X-Shift={x_shift:.3f})! Fallback to random.")
                    indices = np.random.choice(10, 2, replace=False)
                    l_idx, r_idx = indices[0], indices[1]
                else:
                    if args.get('pretrain_mode') == 'cooperative':
                        # Cooperative Proximity Constraint: Max 2 objects between G and B (idx diff <= 3)
                        possible_pairs = []
                        for li in left_candidates:
                            for ri in right_candidates:
                                if li != ri:
                                    possible_pairs.append((li, ri))
                        
                        # Filter by proximity
                        proximate_pairs = [p for p in possible_pairs if abs(p[0] - p[1]) <= 3]
                        
                        if proximate_pairs:
                            pair_idx = np.random.choice(len(proximate_pairs))
                            l_idx, r_idx = proximate_pairs[pair_idx]
                            print(f"  [Gen] Cooperative Selection (Proximate). Pairs found: {len(proximate_pairs)}")
                        else:
                            # Fallback: Pick the pair with the smallest distance
                            possible_pairs.sort(key=lambda p: abs(p[0] - p[1]))
                            l_idx, r_idx = possible_pairs[0]
                            print(f"  [Gen] WARNING: No proximate pairs found. Falling back to closest pair (diff={abs(l_idx - r_idx)})")
                    else:
                        # Independent mode: Random selection from candidates
                        l_idx = np.random.choice(left_candidates)
                        # Ensure r_idx is different from l_idx
                        r_candidates_filtered = [i for i in right_candidates if i != l_idx]
                        if not r_candidates_filtered:
                            r_idx = np.random.choice(right_candidates) # Fallback to same if unique selection impossible (rare)
                        else:
                            r_idx = np.random.choice(r_candidates_filtered)
                    
                    indices = np.array([l_idx, r_idx])
                
                print(f"  [Gen] Task Selection. Left: {l_idx} (x={xs[l_idx]:.3f}), Right: {r_idx} (x={xs[r_idx]:.3f})")

                if args.get('pretrain_mode') == 'independent':
                    # Independent: Commands = I
                    command_sequence_str = "I"
                    command_queue_template = list(command_sequence_str)
                
                    # Base: random g/b for all
                    c_list = np.random.choice(['g', 'b'], size=10).tolist()
                
                    c_list[indices[0]] = 'r'
                    c_list[indices[1]] = 'r'
                
                    color_seq = c_list

                elif args.get('pretrain_mode') == 'cooperative':
                    # Cooperative: Only one pair, so use "C"
                    command_sequence_str = "C"
                    command_queue_template = list(command_sequence_str)
                
                    # Colors: The rest are 'r'
                    c_list = ['r'] * 10
                
                    # Decide which is G and which is B
                    pair = ['g', 'b']
                    np.random.shuffle(pair)
                
                    c_list[indices[0]] = pair[0]
                    c_list[indices[1]] = pair[1]
                
                    color_seq = c_list
            
                # Inject for pretrain mode
                MANYCUBES_COLORS[0] = color_seq
                MANYCUBES_TASK_COUNT[0] = len(command_queue_template)
                MANYCUBES_CONFIG['target_indices'] = set(indices.tolist())
            
            else:
                # Normal mode: use provided command sequence and color sequence
                # command_queue_template and color_seq are already set from args
                MANYCUBES_COLORS[0] = color_seq if color_seq is not None else COLOR_SEQUENCE
                MANYCUBES_TASK_COUNT[0] = len(command_queue_template)
                MANYCUBES_CONFIG['target_indices'] = None
        
            # Force EGL for headless rendering during rollout (if needed, or just let it be)
            # os.environ['MUJOCO_GL'] = 'egl' 
        
            # EE environment needs cameras only for onscreen rendering
            ee_cameras = camera_names if onscreen_render else []
            env = make_ee_sim_env(task_name, camera_names=ee_cameras) 
            ts = env.reset()
            episode = [ts]
        
            policy = InteractivePolicy(inject_noise=inject_noise, color_sequence=color_seq)
            policy.init_pose(ts)

            if onscreen_render:
                 import cv2
                 window_name = "Dataset Generation"
                 cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
                 try:
                     cv2.startWindowThread()
                 except:
                     pass

        
        
            # Schedule all commands upfront (new async system)
            for cmd in command_queue_template:
                policy.schedule_command(cmd, ts)
        
            # Loop for EE rollout
            # Use a large max_steps to allow dynamic termination
            max_steps = 3000 
            for step in range(max_steps):
                # Process command buffer (async execution)
                policy.process_command_buffer(ts)
            
                # Dynamic termination: if buffer is empty and both arms are free, done
                if not policy.command_buffer:
                    left_free = policy.is_arm_free(True, step)
                    right_free = policy.is_arm_free(False, step)
                    if left_free and right_free:
                        print(f"  [EE] All tasks completed. Terminating at step {step}")
                        break # Break from loop, then finalize and print

                action = policy(ts)
                ts = env.step(action)
                episode.append(ts)
            
                # Optional: Render
                if onscreen_render:
                    # Render every 5 steps (reduced from 2)
                    if step % 5 == 0:
                        img_rgb = ts.observation['images']['top']
                        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                        # Draw vertical line at center (x=160) to distinguish left/right
                        cv2.line(img_bgr, (160, 0), (160, 240), (0, 255, 0), 1)
                        cv2.imshow(window_name, img_bgr)
                        cv2.waitKey(1)
 

            # Finalize any open segments in the policy metadata
            policy.finalize(step)
            print(f"  [EE] Episode finished at step {step}")

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
            # for j in range(14):
            #      joint_traj_np[:, j] = np.unwrap(joint_traj_np[:, j])
        
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

            # Save task segment metadata before deleting policy
            left_segments = policy.left_segments.copy()
            right_segments = policy.right_segments.copy()

            del env
            del policy

            # ---------------------------------------------------------
            # 2. Joint Space Replay
            # ---------------------------------------------------------
            print(f'Episode {episode_idx}: Replaying in Joint space')
            print(f'  EE Episode Length: {len(episode)} steps')
            print(f'  Joint Trajectory Length: {len(joint_traj)} steps')
        
            if len(episode) != len(joint_traj):
                print(f'  WARNING: Length mismatch! EE={len(episode)}, Joint={len(joint_traj)}')
        
            # Inject Initial Poses from EE Rollout
            poses = {}
            for i in range(10):
                poses[i] = subtask_info[i*7 : (i+1)*7].copy()
            MANYCUBES_POSES[0] = poses
            MANYCUBES_COLORS[0] = color_seq if color_seq is not None else COLOR_SEQUENCE
            MANYCUBES_TASK_COUNT[0] = len(command_queue_template)
        
            # Create replay environment
            # Only disable cameras if we're not saving data (optimization)
            if args['no_save_data']:
                replay_cameras = ['top'] if onscreen_render else []
            else:
                replay_cameras = camera_names  # Need cameras for dataset images
            env = make_sim_env(task_name, camera_names=replay_cameras, time_limit=2000) # 2000s is plenty
            ts = env.reset()
            episode_replay = [ts]
        
            # Cache joint IDs for object removal
            cube_joint_ids = {}
            for i in range(10):
                try:
                    cube_joint_ids[i] = env.physics.model.name2id(f'cube_{i}_joint', 'joint')
                except:
                    cube_joint_ids[i] = None

            removal_timers = {}
            removed_objects = set()
            DT = 0.02 # From piper_constants

            # Replay Loop
            import time as time_module
            replay_start = time_module.time()
            step_times = []
            max_reward_achieved = 0  # Track maximum reward during replay
        
            for t in range(len(joint_traj)):
                t_start = time_module.time()
                action = joint_traj[t].copy()
                ts = env.step(action)
                episode_replay.append(ts)
                step_times.append(time_module.time() - t_start)
            
                # Track maximum reward achieved
                if ts.reward is not None:
                    max_reward_achieved = max(max_reward_achieved, ts.reward)
            
                if onscreen_render:
                    if t % 5 == 0:
                        img_rgb = ts.observation['images'][render_cam_name]
                        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                        # Draw vertical line at center (x=160) to distinguish left/right
                        cv2.line(img_bgr, (160, 0), (160, 240), (0, 255, 0), 1)
                        cv2.imshow(window_name, img_bgr)
                        cv2.waitKey(1)

                # Object Removal Logic (Mirroring piper_ee_sim_env.py after_step)
                current_time = t * DT
                physics = env.physics
            
                for i in range(10):
                    if i in removed_objects:
                        continue
                    
                    joint_id = cube_joint_ids.get(i)
                    if joint_id is None:
                        continue
                        
                    try:
                        qpos_adr = physics.model.jnt_qposadr[joint_id]
                        y_pos = physics.data.qpos[qpos_adr + 1]
                    
                        if y_pos < 0.28: # In Placement Zone
                            if i not in removal_timers:
                                removal_timers[i] = current_time
                        
                            if current_time - removal_timers[i] > 4.0:
                                # Remove (Teleport)
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
        
            replay_time = time_module.time() - replay_start
            avg_step_time = np.mean(step_times) * 1000
            print(f"  Replay completed in {replay_time:.1f}s ({avg_step_time:.1f}ms/step)")

            # Calculate expected max reward from command sequence
            # I (Independent): 2 points (2 red cubes)
            # C (Cooperative): 2 points (1 green-blue pair)
            # L (Left Independent): 1 point (1 red cube)
            # R (Right Independent): 1 point (1 red cube)
            # T (Transfer): 2 points (cooperative variant)
            independent_i_count = command_queue_template.count('I')  # 2 points each
            independent_lr_count = command_queue_template.count('L') + command_queue_template.count('R')  # 1 point each
            cooperative_count = command_queue_template.count('C') + command_queue_template.count('T')  # 2 points each
            expected_max_reward = 2 * independent_i_count + 1 * independent_lr_count + 2 * cooperative_count
            
            print(f"  Max reward achieved: {max_reward_achieved} / {expected_max_reward} expected")

            # Verify Success based on:
            # 1. Metadata existence (segments)
            # 2. Max reward achieved
            if left_segments or right_segments:
                if max_reward_achieved >= expected_max_reward:
                    print(f"Episode {episode_idx} Successful (Metadata generated, Max reward achieved)")
                    episode_success = True
                    success_count += 1
                else:
                    print(f"Episode {episode_idx} Failed (Max reward not achieved: {max_reward_achieved}/{expected_max_reward})")
                    retry_count += 1
                    total_retries += 1
            else:
                print(f"Episode {episode_idx} Failed (No metadata generated)")
                retry_count += 1
                total_retries += 1
            
            if not episode_success:
                if retry_count >= max_retries_per_episode:
                    print(f"Episode {episode_idx} FAILED after {max_retries_per_episode} attempts. Skipping.")
                    failed_sequences_count += 1
                    # Skip this index and move on
                    episode_idx += 1
                    break 
                else:
                    print(f"Retrying episode {episode_idx}...")
                    # Clean up before retry
                    del env
                    del episode_replay
                    continue  # Retry the episode
        
        # Sequence File Status Update
        if args.get('sequence_file'):
            import csv
            seq_file = args['sequence_file']
            
            # Update the row status
            # We can reuse 'rows' and 'seq_idx' from the start of the loop
            # Ensure we have the latest file content if needed, but since we are the only writer (presumably),
            # memory 'rows' should be mostly fine. But safe to re-read if multiple processes?
            # For simplicity/safety in single process, use memory.
            # Actually, to be robust against manual edits or parallel, let's re-read?
            # Re-reading is safer. Find the row by content? Or just index if file hasn't changed.
            # Let's assume index is stable for this process step.
            
            # Check success
            if args.get('test'):
                status = 'test'
            else:
                status = 'success' if episode_success else 'failed'
            
            # Update
            if len(rows[seq_idx]) < 3:
                    rows[seq_idx].append(status)
            else:
                    rows[seq_idx][2] = status
            
            # Write back
            with open(seq_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(rows)

        # Only save if successful
        if not episode_success:
            print(f"Episode {episode_idx} (Sequence {seq_idx if args.get('sequence_file') else '?'}) FAILED. Skipping save.")
            continue

            # ---------------------------------------------------------
            # 3. Save to HDF5
            # ---------------------------------------------------------
        
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
            ts = episode_replay[k]  # Use Joint replay observations (including images)
        
            data_dict['/observations/qpos'].append(ts.observation['qpos'])
            data_dict['/observations/qvel'].append(ts.observation['qvel'])
            data_dict['/action'].append(action)
            for cam_name in camera_names:
                if cam_name in ts.observation['images']:
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
                    _ = image.create_dataset(cam_name, (num_steps_to_save, 240, 320, 3), dtype='uint8',
                                             chunks=(1, 240, 320, 3), )
                qpos = obs.create_dataset('qpos', (num_steps_to_save, 14))
                qvel = obs.create_dataset('qvel', (num_steps_to_save, 14))
                action = root.create_dataset('action', (num_steps_to_save, 14))
            
                # Save task segment metadata
                metadata = root.create_group('metadata')
                # Convert segment lists to structured array with new fields
                if left_segments:
                    left_seg_data = np.array([
                        (s['start'], s['end'], s['type'], 
                         s.get('top_arm', ''), s.get('coop_split') if s.get('coop_split') is not None else -1)
                        for s in left_segments
                    ], dtype=[('start', 'i4'), ('end', 'i4'), ('type', 'S16'), 
                             ('top_arm', 'S8'), ('coop_split', 'i4')])
                    metadata.create_dataset('left_segments', data=left_seg_data)
                if right_segments:
                    right_seg_data = np.array([
                        (s['start'], s['end'], s['type'],
                         s.get('top_arm', ''), s.get('coop_split') if s.get('coop_split') is not None else -1)
                        for s in right_segments
                    ], dtype=[('start', 'i4'), ('end', 'i4'), ('type', 'S16'),
                             ('top_arm', 'S8'), ('coop_split', 'i4')])
                    metadata.create_dataset('right_segments', data=right_seg_data)

                for name, array in data_dict.items():
                    root[name][...] = array
            print(f'Saving: {time.time() - t0:.1f} secs\n')

        del env
        del episode_replay
    
        print(f"DEBUG: Incrementing episode_idx from {episode_idx} to {episode_idx+1}")
        episode_idx += 1

    print(f'Done. Saved to {dataset_dir}')
    print(f'Success: {success_count}/{num_episodes} episodes')
    print(f'Total retries: {total_retries}')
    print(f'Failed sequences: {failed_sequences_count}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', action='store', type=str, help='task_name', default='sim_many_cubes')
    parser.add_argument('--dataset_dir', action='store', type=str, help='dataset saving dir', required=True)
    parser.add_argument('--num_episodes', action='store', type=int, help='num_episodes', required=True)
    parser.add_argument('--commands', action='store', type=str, help='Command sequence (e.g. ICI)', default=None)
    parser.add_argument('--pretrain_mode', action='store', type=str, choices=['independent', 'cooperative'], help='Pre-training mode to auto-generate patterns')
    parser.add_argument('--sequence_file', action='store', type=str, help='CSV file with sequences and commands', default=None)
    parser.add_argument('--color_sequence', action='store', type=str, help='Color sequence (e.g. rrggbb for 10 objects)', default=None)
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--no_save_data', action='store_true', help='Do not save HDF5 data')
    parser.add_argument('--test', action='store_true', help='Use unused sequences and mark as test')
    
    args = parser.parse_args()
    main(vars(args))
