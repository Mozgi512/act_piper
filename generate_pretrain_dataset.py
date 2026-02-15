
import time
import os
os.environ['MUJOCO_GL'] = 'egl'

import numpy as np
import argparse
import matplotlib.pyplot as plt
import h5py
import collections
import csv

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

    def append_success_init_positions(csv_path, episode_id, color_sequence, command_queue, env_state_70):
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        row = {
            'episode': int(episode_id),
            'commands': ''.join(command_queue) if command_queue is not None else '',
            'color_sequence': ''.join(color_sequence) if color_sequence is not None else '',
        }
        cube_state = np.array(env_state_70).reshape(10, 7)
        for i in range(10):
            row[f'cube{i}_x'] = float(cube_state[i, 0])
            row[f'cube{i}_y'] = float(cube_state[i, 1])
            row[f'cube{i}_z'] = float(cube_state[i, 2])

        fieldnames = ['episode', 'commands', 'color_sequence']
        for i in range(10):
            fieldnames.extend([f'cube{i}_x', f'cube{i}_y', f'cube{i}_z'])

        file_exists = os.path.isfile(csv_path)
        with open(csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    def is_home_at_episode_end(replay_steps):
        if replay_steps is None or len(replay_steps) < 2:
            return False, None, None
        q_home = np.array(replay_steps[0].observation['qpos']).copy()
        q_last = np.array(replay_steps[-1].observation['qpos']).copy()

        left_arm_diff = np.max(np.abs(q_last[:6] - q_home[:6]))
        right_arm_diff = np.max(np.abs(q_last[7:13] - q_home[7:13]))
        arm_diff = float(max(left_arm_diff, right_arm_diff))
        grip_diff = float(max(abs(q_last[6] - q_home[6]), abs(q_last[13] - q_home[13])))

        ok = (arm_diff <= float(args.get('end_home_arm_threshold', 0.20))) and \
             (grip_diff <= float(args.get('end_home_gripper_threshold', 0.35)))
        return ok, arm_diff, grip_diff

    def set_cube_alpha(physics, alpha_value):
        for gid in range(physics.model.ngeom):
            gname = physics.model.id2name(gid, 'geom')
            if gname is None:
                continue
            if gname == 'cube' or gname.startswith('cube_'):
                physics.model.geom_rgba[gid, 3] = float(alpha_value)
        physics.forward()

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

    if not args.get('pretrain_mode') and not args.get('wait_only_mode') and not command_sequence_str:
        parser.error("--commands is required unless --pretrain_mode or --wait_only_mode is specified.")


    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir, exist_ok=True)

    episode_len = SIM_TASK_CONFIGS[task_name]['episode_len']
    camera_names = SIM_TASK_CONFIGS[task_name]['camera_names']

    success_count = 0
    episode_idx = 0
    total_retries = 0
    max_retries_per_episode = 5
    
    while episode_idx < num_episodes:
        retry_count = 0
        episode_success = False
        
        while retry_count < max_retries_per_episode and not episode_success:
            independent_right_target_idx = None
            if retry_count > 0:
                print(f'Episode {episode_idx}: Retry attempt {retry_count}/{max_retries_per_episode}')
            else:
                print(f'Episode {episode_idx}: Rollout out EE space InteractivePolicy')
        
            # ---------------------------------------------------------
            # 1. EE Rollout
            # ---------------------------------------------------------
        

            # Global Random X-Shift (0.0 to 0.10)
            # Combined with base shift 0.02, total shift is 0.02 to 0.12
            # NOTE: This randomization occurs on EVERY retry attempt
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
            
            # Reset target_indices
            MANYCUBES_CONFIG['target_indices'] = None
            # Reset target placement jitter to legacy defaults (used by piper_ee_sim_env)
            MANYCUBES_CONFIG['target_jitter_x'] = 0.04
            MANYCUBES_CONFIG['target_y_min'] = 0.32
            MANYCUBES_CONFIG['target_y_max'] = 0.45

            # Pre-training Mode Logic
            if args.get('pretrain_mode'):
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
                        # Cooperative selection with reduced index bias:
                        # sample one anchor arm uniformly, then sample the other arm conditionally.
                        # Apply proximity preference (idx diff <= 3) probabilistically to avoid sparse regions.
                        max_abs_dx = float(args.get('pretrain_coop_max_abs_dx', 0.70))
                        jitter_x_cfg = float(args.get('pretrain_coop_target_jitter_x', 0.08))
                        # Conservative center-distance limit so sampled jitter still tends to satisfy max |dx|.
                        center_dx_limit = max(0.0, max_abs_dx - 2.0 * jitter_x_cfg)

                        use_proximity = (np.random.rand() < float(args.get('pretrain_coop_proximate_prob', 0.6)))
                        anchor_left = bool(np.random.rand() < 0.5)
                        if anchor_left:
                            l_idx = int(np.random.choice(left_candidates))
                            prox_right = [ri for ri in right_candidates if ri != l_idx and abs(l_idx - ri) <= 3]
                            valid_right = [ri for ri in right_candidates if ri != l_idx and abs(xs[l_idx] - xs[ri]) <= center_dx_limit]
                            prox_right_valid = [ri for ri in prox_right if abs(xs[l_idx] - xs[ri]) <= center_dx_limit]
                            if use_proximity and prox_right:
                                pick_pool = prox_right_valid if prox_right_valid else prox_right
                                r_idx = int(np.random.choice(pick_pool))
                            else:
                                right_pool = [ri for ri in right_candidates if ri != l_idx] or right_candidates
                                pick_pool = valid_right if valid_right else right_pool
                                r_idx = int(np.random.choice(pick_pool))
                        else:
                            r_idx = int(np.random.choice(right_candidates))
                            prox_left = [li for li in left_candidates if li != r_idx and abs(li - r_idx) <= 3]
                            valid_left = [li for li in left_candidates if li != r_idx and abs(xs[li] - xs[r_idx]) <= center_dx_limit]
                            prox_left_valid = [li for li in prox_left if abs(xs[li] - xs[r_idx]) <= center_dx_limit]
                            if use_proximity and prox_left:
                                pick_pool = prox_left_valid if prox_left_valid else prox_left
                                l_idx = int(np.random.choice(pick_pool))
                            else:
                                left_pool = [li for li in left_candidates if li != r_idx] or left_candidates
                                pick_pool = valid_left if valid_left else left_pool
                                l_idx = int(np.random.choice(pick_pool))

                        # Final safeguard: if still too far in center-distance, move counterpart to nearest valid.
                        center_dx = abs(xs[l_idx] - xs[r_idx])
                        if center_dx > center_dx_limit:
                            if anchor_left:
                                valid_right = [ri for ri in right_candidates if ri != l_idx and abs(xs[l_idx] - xs[ri]) <= center_dx_limit]
                                if valid_right:
                                    r_idx = min(valid_right, key=lambda ri: abs(xs[l_idx] - xs[ri]))
                            else:
                                valid_left = [li for li in left_candidates if li != r_idx and abs(xs[li] - xs[r_idx]) <= center_dx_limit]
                                if valid_left:
                                    l_idx = min(valid_left, key=lambda li: abs(xs[li] - xs[r_idx]))
                            center_dx = abs(xs[l_idx] - xs[r_idx])

                        print(
                            f"  [Gen] Cooperative Selection (anchor uniform, prox_prob={args.get('pretrain_coop_proximate_prob', 0.6):.2f}). "
                            f"anchor={'L' if anchor_left else 'R'} use_proximity={use_proximity} center|dx|={center_dx:.3f} limit={center_dx_limit:.3f}"
                        )
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
                    independent_right_target_idx = int(r_idx)
                
                    # Base: random g/b for all
                    c_list = np.random.choice(['g', 'b'], size=10).tolist()
                
                    c_list[indices[0]] = 'r'
                    c_list[indices[1]] = 'r'
                
                    color_seq = c_list

                elif args.get('pretrain_mode') == 'cooperative':
                    # Cooperative: Only one pair, so use "C"
                    command_sequence_str = "C"
                    command_queue_template = list(command_sequence_str)

                    # Increase cooperative position diversity
                    MANYCUBES_CONFIG['target_jitter_x'] = float(args.get('pretrain_coop_target_jitter_x', 0.08))
                    MANYCUBES_CONFIG['target_y_min'] = float(args.get('pretrain_coop_target_y_min', 0.32))
                    MANYCUBES_CONFIG['target_y_max'] = float(args.get('pretrain_coop_target_y_max', 0.45))
                
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
                print(f"  [Gen] Task Target Indices: {MANYCUBES_CONFIG['target_indices']}")
            
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

            if onscreen_render:
                 import cv2
                 window_name = "Dataset Generation"
                 cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
                 try:
                     cv2.startWindowThread()
                 except:
                     pass

            if args.get('wait_only_mode'):
                wait_steps = int(args.get('wait_only_steps', 200))
                set_cube_alpha(env.physics, 0.0)
                hold_action = np.concatenate([
                    ts.observation['mocap_pose_left'],
                    np.array([1.0]),
                    ts.observation['mocap_pose_right'],
                    np.array([1.0]),
                ])
                for step in range(wait_steps):
                    ts = env.step(hold_action)
                    episode.append(ts)
                left_segments = [{'start': 0, 'end': wait_steps, 'type': 'independent'}]
                right_segments = [{'start': 0, 'end': wait_steps, 'type': 'independent'}]
                print(f"  [EE] Wait-only rollout finished at step {wait_steps}")
            else:
                policy = InteractivePolicy(inject_noise=inject_noise, color_sequence=color_seq)
                policy.init_pose(ts)

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

                # Save task segment metadata before deleting policy
                left_segments = policy.left_segments.copy()
                right_segments = policy.right_segments.copy()

            # Capture Env State for Replay
            # env_state includes robot state + object poses
            # For sim_many_cubes: 
            # qpos structure: robot(16) + belt(1) + belt_ext(1) + 10 cubes (7*10) = 88
            # get_env_state returns qpos[18:18+70] -> The 10 cubes.
            subtask_info = episode[0].observation['env_state'].copy() 

            if args.get('pretrain_mode') == 'independent' and independent_right_target_idx is not None:
                right_init_x = float(subtask_info[independent_right_target_idx * 7 + 0])
                if right_init_x < 0.0:
                    print(
                        f"Episode {episode_idx} Failed (Independent right target x<0: "
                        f"idx={independent_right_target_idx}, x={right_init_x:.4f})"
                    )
                    retry_count += 1
                    total_retries += 1
                    del env
                    if not args.get('wait_only_mode'):
                        del policy
                    if retry_count >= max_retries_per_episode:
                        print(f"Episode {episode_idx} FAILED after {max_retries_per_episode} attempts. Skipping.")
                        episode_idx += 1
                        break
                    else:
                        print(f"Retrying episode {episode_idx}...")
                        continue
        
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

            del env
            if not args.get('wait_only_mode'):
                del policy

            # ---------------------------------------------------------
            # 2. Joint Space Replay
            # ---------------------------------------------------------
            print(f'Episode {episode_idx}: Replaying in Joint space')
            print(f'  EE Episode Length: {len(episode)} steps')
            print(f'  Joint Trajectory Length: {len(joint_traj)} steps')

            data_steps = max(0, len(joint_traj) - 1)
            max_data_steps = args.get('max_data_steps', 0)
            if max_data_steps and data_steps > int(max_data_steps):
                print(f"Episode {episode_idx} Failed (Data steps exceeded: {data_steps}>{int(max_data_steps)})")
                retry_count += 1
                total_retries += 1
                if retry_count >= max_retries_per_episode:
                    print(f"Episode {episode_idx} FAILED after {max_retries_per_episode} attempts. Skipping.")
                    episode_idx += 1
                    break
                else:
                    print(f"Retrying episode {episode_idx}...")
                    continue
        
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
            env = make_sim_env(task_name, camera_names=replay_cameras, time_limit=2000)
            ts = env.reset()
            if args.get('wait_only_mode'):
                set_cube_alpha(env.physics, 0.0)
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

            end_home_ok, end_arm_diff, end_grip_diff = is_home_at_episode_end(episode_replay)
            print(
                f"  End-home check: ok={end_home_ok} arm_diff={end_arm_diff:.4f} "
                f"grip_diff={end_grip_diff:.4f} "
                f"(th_arm={args.get('end_home_arm_threshold', 0.20):.3f}, "
                f"th_grip={args.get('end_home_gripper_threshold', 0.35):.3f})"
            )

            # Verify Success based on:
            # 1. Metadata existence (segments)
            # 2. Max reward achieved
            if left_segments or right_segments:
                if max_reward_achieved >= expected_max_reward and end_home_ok:
                    print(f"Episode {episode_idx} Successful (Metadata generated, Max reward achieved, End-home satisfied)")
                    episode_success = True
                    success_count += 1
                else:
                    if max_reward_achieved < expected_max_reward:
                        print(f"Episode {episode_idx} Failed (Max reward not achieved: {max_reward_achieved}/{expected_max_reward})")
                    else:
                        print(f"Episode {episode_idx} Failed (End-home not satisfied)")
                    retry_count += 1
                    total_retries += 1
            else:
                print(f"Episode {episode_idx} Failed (No metadata generated)")
                retry_count += 1
                total_retries += 1
            
                if retry_count >= max_retries_per_episode:
                    print(f"Episode {episode_idx} FAILED after {max_retries_per_episode} attempts. Skipping.")
                    # Skip this episode entirely, move to next
                    episode_idx += 1
                    break
                else:
                    print(f"Retrying episode {episode_idx}...")
                    # Clean up before retry
                    del env
                    del episode_replay
                    continue  # Retry the episode
        
            # Only save if successful
            if not episode_success:
                if args.get('fail_fast_on_episode_failure'):
                    print(f"Fail-fast: episode {episode_idx} failed. Early termination.")
                    return
                continue

            # Save successful episode initial object positions to CSV
            init_pos_csv_path = args.get('save_init_positions_csv')
            if not init_pos_csv_path:
                init_pos_csv_path = os.path.join(dataset_dir, 'successful_init_positions.csv')
            append_success_init_positions(
                init_pos_csv_path,
                episode_idx,
                color_seq if color_seq is not None else COLOR_SEQUENCE,
                command_queue_template,
                subtask_info,
            )

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
        
            episode_idx += 1

    print(f'Done. Saved to {dataset_dir}')
    print(f'Success: {success_count}/{num_episodes} episodes')
    print(f'Total retries: {total_retries}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', action='store', type=str, help='task_name', default='sim_many_cubes')
    parser.add_argument('--dataset_dir', action='store', type=str, help='dataset saving dir', required=True)
    parser.add_argument('--num_episodes', action='store', type=int, help='num_episodes', required=True)
    parser.add_argument('--commands', action='store', type=str, help='Command sequence (e.g. ICI)', default=None)
    parser.add_argument('--pretrain_mode', action='store', type=str, choices=['independent', 'cooperative'], help='Pre-training mode to auto-generate patterns')
    parser.add_argument('--wait_only_mode', action='store_true', help='Generate wait-only data: no visible objects and no task execution')
    parser.add_argument('--wait_only_steps', action='store', type=int, default=200, help='Number of steps for wait-only rollout')
    parser.add_argument('--pretrain_coop_target_jitter_x', action='store', type=float, default=0.08, help='Per-target X jitter amplitude for cooperative pretrain placement')
    parser.add_argument('--pretrain_coop_target_y_min', action='store', type=float, default=0.32, help='Min Y for cooperative pretrain target placement')
    parser.add_argument('--pretrain_coop_target_y_max', action='store', type=float, default=0.45, help='Max Y for cooperative pretrain target placement')
    parser.add_argument('--pretrain_coop_proximate_prob', action='store', type=float, default=0.6, help='Probability of enforcing cooperative proximity constraint (|idx diff|<=3)')
    parser.add_argument('--pretrain_coop_max_abs_dx', action='store', type=float, default=0.70, help='Approximate max |Δx| for cooperative pair centers (conservative with jitter margin)')
    parser.add_argument('--max_data_steps', action='store', type=int, default=0, help='If >0, fail an episode attempt when data step count exceeds this limit')
    parser.add_argument('--fail_fast_on_episode_failure', action='store_true', help='Terminate immediately when an episode fails instead of continuing')
    parser.add_argument('--color_sequence', action='store', type=str, help='Color sequence (e.g. rrggbb for 10 objects)', default=None)
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--no_save_data', action='store_true', help='Do not save HDF5 data')
    parser.add_argument('--end_home_arm_threshold', action='store', type=float, default=0.20, help='Max allowed arm joint deviation from start pose at episode end')
    parser.add_argument('--end_home_gripper_threshold', action='store', type=float, default=0.35, help='Max allowed gripper deviation from start pose at episode end')
    parser.add_argument('--save_init_positions_csv', action='store', type=str, default=None, help='Optional path to save successful episodes initial object positions CSV')
    
    args = parser.parse_args()
    main(vars(args))
