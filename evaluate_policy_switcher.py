import sys
import os
import tty
import termios
import select
import time
from copy import deepcopy
import pickle
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
from einops import rearrange
import IPython
import cv2

# Add detr to path
cwd = os.getcwd()
sys.path.append(os.path.join(cwd, 'detr'))

# Import existing modules
from piper_constants import DT
from piper_constants import PUPPET_GRIPPER_JOINT_OPEN
from utils import load_data
from utils import sample_redbox_pose, sample_insertion_pose, sample_greenbox_pose, sample_bluebox_pose
from utils import compute_dict_mean, set_seed, detach_dict
from policy import ACTPolicy

# Import Sim Env
from piper_sim_env import REDBOX_POSE, GREENBOX_POSE, BLUEBOX_POSE
from piper_sim_env import make_sim_env, MANYCUBES_COLORS, MANYCUBES_TASK_COUNT
from piper_ee_sim_env import make_ee_sim_env
from utils import apply_rgb_mask_to_strip, apply_rgb_mask_to_right_strip
# Constants
MODE_INDEPENDENT = '1'
MODE_COOP = '2'

COLOR_SEQUENCE = ['r', 'r', 'g', 'b', 'r', 'r', 'g', 'b', 'r', 'g']

def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy

def get_image_dual(ts, camera_names):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(ts.observation['images'][cam_name], 'h w c -> c h w')
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_image

def get_image_independent(ts, camera_names, arm):
    curr_images = []
    for cam_name in camera_names:
        # Original: H W C
        curr_image = ts.observation['images'][cam_name]
        h, w, _ = curr_image.shape
        
        # Calculate offset based on width (base 640 -> 40)
        offset = int(40 * (w / 640))

        if arm == 'left':
            # Left Arm: masks the RIGHT side (overlap region)
            curr_image = curr_image[:, :w//2, :].copy()
            curr_image = apply_rgb_mask_to_right_strip(curr_image, strip_width=offset)
        else:
            # Shift based on width (adaptive)
            start = w//2 - offset
            end = w - offset
            # Right Arm: masks the LEFT side (overlap region)
            curr_image = curr_image[:, start:end, :].copy()
            
            # Apply RGB mask
            curr_image = apply_rgb_mask_to_strip(curr_image, strip_width=offset)
            
        curr_image = rearrange(curr_image, 'h w c -> c h w')
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_image

def remove_cubes(physics, indices):
    """Teleport cubes to z=-10 to effectively remove them."""
    for i in indices:
        try:
            start_idx = physics.model.name2id(f'cube_{i}_joint', 'joint')
            qpos_adr = physics.model.jnt_qposadr[start_idx]
            # Set Z to -10, and also move X/Y far away to be safe
            physics.data.qpos[qpos_adr + 0] = 10.0 + i # X far
            physics.data.qpos[qpos_adr + 1] = 10.0 + i # Y far
            physics.data.qpos[qpos_adr + 2] = -10.0    # Z under
        except Exception as e:
            # Indices might go out of range if not initialized, ignore
            pass

def get_touched_cubes(physics):
    """Return set of cube indices that are currently in contact with any gripper."""
    touched = set()
    for i_contact in range(physics.data.ncon):
        id_geom_1 = physics.data.contact[i_contact].geom1
        id_geom_2 = physics.data.contact[i_contact].geom2
        name_1 = physics.model.id2name(id_geom_1, 'geom')
        name_2 = physics.model.id2name(id_geom_2, 'geom')
        
        if name_1 is None or name_2 is None: continue
        
        # Check pair (gripper, cube)
        for n1, n2 in [(name_1, name_2), (name_2, name_1)]:
            if 'gripper' in n1 and 'cube_' in n2:
                # Extract index from 'cube_X' or 'cube_X_g0' etc.
                try:
                    # Assumes name structure 'cube_{idx}...'
                    parts = n2.split('_')
                    # Find 'cube' then next part is index
                    if 'cube' in parts:
                        idx_loc = parts.index('cube') + 1
                        if idx_loc < len(parts):
                            cube_idx = int(parts[idx_loc])
                            touched.add(cube_idx)
                except:
                    pass
    return touched

def load_policy_and_stats(ckpt_dir, policy_class, args, override_state_dim=None, override_arm=None):
    state_dim = 14
    if override_state_dim:
        state_dim = override_state_dim
        
    camera_names = ['top'] 

    lr_backbone = 1e-5
    backbone = 'resnet18'
    
    policy_config = {
        'lr': args.lr,
        'num_queries': args.chunk_size,
        'kl_weight': args.kl_weight,
        'hidden_dim': args.hidden_dim,
        'dim_feedforward': args.dim_feedforward,
        'lr_backbone': lr_backbone,
        'backbone': backbone,
        'enc_layers': 4,
        'dec_layers': 7,
        'nheads': 8,
        'camera_names': camera_names,
        'state_dim': state_dim
    }
    
    if override_arm:
        policy_config['arm'] = override_arm

    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    ckpt_path = os.path.join(ckpt_dir, 'policy_best.ckpt')
    policy = make_policy(policy_class, policy_config)
    loaded_state_dict = torch.load(ckpt_path)
    
    # Handle chunk_size mismatch
    if 'model.pos_table' in loaded_state_dict and 'model.query_embed.weight' in loaded_state_dict:
        curr_pos_len = policy.model.pos_table.shape[1]
        load_pos_len = loaded_state_dict['model.pos_table'].shape[1]
        if load_pos_len > curr_pos_len:
             loaded_state_dict['model.pos_table'] = loaded_state_dict['model.pos_table'][:, :curr_pos_len, :]
             
        curr_query_len = policy.model.query_embed.weight.shape[0]
        load_query_len = loaded_state_dict['model.query_embed.weight'].shape[0]
        if load_query_len > curr_query_len:
             loaded_state_dict['model.query_embed.weight'] = loaded_state_dict['model.query_embed.weight'][:curr_query_len, :]

    policy.load_state_dict(loaded_state_dict)
    policy.cuda()
    policy.eval()
    
    print(f'Loaded policy from {ckpt_dir}')
    
    return policy, stats

def main(args):
    set_seed(1000)
    
    task_name = args.task_name
    ckpt_dual = args.ckpt_dual
    ckpt_left = args.ckpt_left
    ckpt_right = args.ckpt_right
    policy_class = args.policy_class
    onscreen_render = args.onscreen_render

    # Parse Command Sequence
    command_queue = list(args.commands) if args.commands else []
    print(f"Command Sequence: {command_queue}")
    
    # Parse Color Sequence
    color_seq = None
    if args.color_sequence:
        color_seq = list(args.color_sequence.lower())
        if len(color_seq) != 10:
            print(f"Warning: Color sequence should be 10 chars (got {len(color_seq)})")
    
    # Inject Globals
    MANYCUBES_COLORS[0] = color_seq if color_seq is not None else COLOR_SEQUENCE
    MANYCUBES_TASK_COUNT[0] = len(command_queue)
    
    
    from piper_constants import SIM_TASK_CONFIGS
    task_config = SIM_TASK_CONFIGS[task_name]
    # episode_len overridden by dynamic subtask lengths
    # episode_len = task_config['episode_len']
    camera_names = task_config['camera_names']

    camera_names = task_config['camera_names']

    # E2E Mode Logic
    if args.ckpt_e2e:
        print(f"Loading E2E Policy from {args.ckpt_e2e}...")
        policy_e2e, stats_e2e = load_policy_and_stats(args.ckpt_e2e, policy_class, args, override_state_dim=14)
        
        # Alias both modes to E2E policy
        policy_dual = policy_e2e
        stats_dual = stats_e2e
        
        policy_independent_dual = policy_e2e
        stats_independent_dual = stats_e2e
        
        # Dummy placeholders for legacy to avoid errors (though not sure if needed if logic flows right)
        policy_left = None
        stats_left = None
        policy_right = None
        stats_right = None
        
    else:
        # Standard Switching Mode
        print("Loading Dual Policy...")
        policy_dual, stats_dual = load_policy_and_stats(ckpt_dual, policy_class, args, override_state_dim=14)
        
        # Optional: Load Independent Dual Policy matches stats of Dual (Coop) usually? 
        # Or does it have its own stats? Likely its own.
        policy_independent_dual = None
        stats_independent_dual = None
        if args.ckpt_independent_dual:
            print("Loading Independent Dual Policy...")
            policy_independent_dual, stats_independent_dual = load_policy_and_stats(args.ckpt_independent_dual, policy_class, args, override_state_dim=14)
        
        policy_left = None
        stats_left = None
        policy_right = None
        stats_right = None
        
        if not args.ckpt_independent_dual:
            # Legacy mode: Load Left/Right
            if not args.ckpt_left or not args.ckpt_right:
                 raise ValueError("If --ckpt_independent_dual (or --ckpt_e2e) is not specified, --ckpt_left and --ckpt_right are required.")
            print("Loading Left Policy...")
            policy_left, stats_left = load_policy_and_stats(ckpt_left, policy_class, args, override_state_dim=7, override_arm='left')
            print("Loading Right Policy...")
            policy_right, stats_right = load_policy_and_stats(ckpt_right, policy_class, args, override_state_dim=7, override_arm='right')

    pre_process_dual = lambda s_qpos: (s_qpos - stats_dual['qpos_mean']) / stats_dual['qpos_std']
    post_process_dual = lambda a: a * stats_dual['action_std'] + stats_dual['action_mean']
    
    if policy_independent_dual:
         pre_process_independent_dual = lambda s_qpos: (s_qpos - stats_independent_dual['qpos_mean']) / stats_independent_dual['qpos_std']
         post_process_independent_dual = lambda a: a * stats_independent_dual['action_std'] + stats_independent_dual['action_mean']
    
    if policy_left:
        pre_process_left = lambda s_qpos: (s_qpos - stats_left['qpos_mean']) / stats_left['qpos_std']
        post_process_left = lambda a: a * stats_left['action_std'] + stats_left['action_mean']
        
        pre_process_right = lambda s_qpos: (s_qpos - stats_right['qpos_mean']) / stats_right['qpos_std']
        post_process_right = lambda a: a * stats_right['action_std'] + stats_right['action_mean']

    # Convert stats to torch for buffer conversion
    def to_torch(x): return torch.from_numpy(x).float().cuda()
    stats_dual_torch = {k: to_torch(v) for k, v in stats_dual.items()}
    if stats_left:
        stats_left_torch = {k: to_torch(v) for k, v in stats_left.items()}
        stats_right_torch = {k: to_torch(v) for k, v in stats_right.items()}

    # Calculate required time limit
    # Default is 20s (1000 steps). We need more for sequences like ICI (1320 steps).
    total_steps_needed = 0
    for cmd in command_queue:
        if cmd.upper() == 'I': total_steps_needed += args.step_i
        else: total_steps_needed += args.step_c
    
    # Add buffer
    total_steps_needed += 500
    time_limit = total_steps_needed * DT

    # Pass camera_names to avoid rendering default 5 cameras (huge speedup)
    # Pass time_limit to avoid 1000 step reset
    env = make_sim_env(task_name, camera_names=camera_names, time_limit=time_limit, interleave_last_four=args.interleave_objects)
    
    # Initialize render vars
    plt_img = None
    if onscreen_render:
        plt.ion()
        ax = plt.subplot()
        # Initial render to get shape
        obs_init = env.reset().observation['images']['top']
        plt_img = ax.imshow(obs_init)
    
    # State tracking
    chunk_size = args.chunk_size
    
    # Initialize Temporal Aggregation
    temporal_agg = not args.no_temporal_agg
    num_queries = args.chunk_size
    # We need a large buffer, or rolling buffer. Since max_timesteps is dynamic (sum of subtasks), let's make it large enough.
    # Assumed max total steps = 4000 -> Reduced to 3000 to save memory for ICI tasks
    MAX_BUFFER_STEPS = 3000
    if total_steps_needed > MAX_BUFFER_STEPS:
         print(f"Warning: total steps ({total_steps_needed}) > MAX_BUFFER_STEPS ({MAX_BUFFER_STEPS}). Increasing buffer.")
         MAX_BUFFER_STEPS = total_steps_needed + 500

    float_nan = float('nan')
    all_time_actions_dual = torch.full([MAX_BUFFER_STEPS, MAX_BUFFER_STEPS+num_queries, 14], float_nan).cuda()
    all_time_actions_left = torch.full([MAX_BUFFER_STEPS, MAX_BUFFER_STEPS+num_queries, 7], float_nan).cuda()
    all_time_actions_right = torch.full([MAX_BUFFER_STEPS, MAX_BUFFER_STEPS+num_queries, 7], float_nan).cuda()

    episode_count = 0
    success_count = 0
    total_rewards = []
    
    try:
        while episode_count < args.num_rollouts:
            # Reset Environment
            ts = env.reset()
            t = 0
            # Get max reward for this episode configuration
            max_possible_reward = env._task.max_reward
            current_reward = 0
            
            # Reset buffers
            all_time_actions_dual.fill_(float_nan)
            all_time_actions_left.fill_(float_nan)
            all_time_actions_right.fill_(float_nan)
            
            step_in_chunk = 0
            current_action_chunk_dual = None
            current_action_chunk_left = None
            current_action_chunk_right = None
            # If unified dual policy is used, we reuse the dual chunk variable? 
            # Or simpler: keep using current_action_chunk_dual for any dual policy (Coop or Indep)
            
            video_frames = []
            
            # Execution Queue
            current_queue = list(command_queue)
            current_subtask_end = -1
            current_mode = None # 'I' or 'C'
            current_cube_idx = 0 # Track which cubes are being processed (2 per task)
            current_touched_cubes = set() # Track cubes touched during this subtask
            
            print(f"\nEpisode {episode_count} Started. Queue: {current_queue}")
            
            # Initial Task Setup
            if current_queue:
                cmd = current_queue.pop(0).upper()
                if cmd == 'I':
                    current_mode = MODE_INDEPENDENT
                    current_subtask_end = t + args.step_i
                else:
                    current_mode = MODE_COOP
                    current_subtask_end = t + args.step_c
                print(f"[Step {t}] Starting First Task: {cmd} (End: {current_subtask_end})")
            else:
                print("No commands provided.")
                break

            while True:
                # -------------------------------
                # Auto-Switching Logic
                # -------------------------------
                if t >= current_subtask_end:
                    # Subtask Finished
                    old_mode = current_mode
                    
                    # Remove used cubes (based on contact history)
                    if len(current_touched_cubes) > 0:
                        print(f"[Step {t}] Subtask finished. Removing touched cubes: {current_touched_cubes}")
                        remove_cubes(env.physics, list(current_touched_cubes))
                    else:
                        print(f"[Step {t}] Subtask finished. No cubes touched.")
                    
                    # Clear for next task
                    current_touched_cubes = set()
                    
                    if current_queue:
                        cmd = current_queue.pop(0).upper()
                        if cmd == 'I':
                            current_mode = MODE_INDEPENDENT
                            current_subtask_end = t + args.step_i
                        else:
                            current_mode = MODE_COOP
                            current_subtask_end = t + args.step_c
                        
                        print(f"[Step {t}] Switching to Task: {cmd} (End: {current_subtask_end})")
                        step_in_chunk = 0 # Force replan on switch
                        
                        # Buffer Inheritance Logic (Smoother transitions)
                        if args.inherit_temporal_buffer and temporal_agg:
                            # Optimize: Only process relevant time window [t : t+num_queries]
                            # Processing entire buffer causes OOM (5000x5100x14)
                            start_col = t
                            end_col = min(t + args.chunk_size, all_time_actions_dual.shape[1])
                            
                            # Case 1: Dual <-> Dual (Unified Indep Policy)
                            # Implicit inheritance via shared buffer 'all_time_actions_dual'.
                            # No explicit copy needed!
                            if policy_independent_dual:
                                 # print("DEBUG: Implicit buffer inheritance for Dual <-> Dual switch")
                                 pass
                            
                            # Case 2: Legacy Split Policies
                            elif old_mode == MODE_COOP and current_mode == MODE_INDEPENDENT:
                                # Dual -> Indep
                                # Slice only relevant columns
                                input_slice = all_time_actions_dual[:, start_col:end_col, :]
                                dual_mask_slice = ~torch.isnan(input_slice)
                                
                                input_slice_safe = torch.nan_to_num(input_slice, nan=0.0)
                                denorm_slice = input_slice_safe * stats_dual_torch['action_std'] + stats_dual_torch['action_mean']
                                
                                # Left
                                denorm_left = denorm_slice[:, :, :7]
                                renorm_left = (denorm_left - stats_left_torch['action_mean']) / stats_left_torch['action_std']
                                val_inherit_left = renorm_left.clone()
                                val_inherit_left[~dual_mask_slice[:, :, :7]] = float('nan')
                                all_time_actions_left[:, start_col:end_col, :].copy_(val_inherit_left)
                                
                                # Right
                                denorm_right = denorm_slice[:, :, 7:]
                                renorm_right = (denorm_right - stats_right_torch['action_mean']) / stats_right_torch['action_std']
                                val_inherit_right = renorm_right.clone()
                                val_inherit_right[~dual_mask_slice[:, :, 7:]] = float('nan')
                                all_time_actions_right[:, start_col:end_col, :].copy_(val_inherit_right)
                                
                            elif old_mode == MODE_INDEPENDENT and current_mode == MODE_COOP:
                                # Indep -> Dual
                                input_left_slice = all_time_actions_left[:, start_col:end_col, :]
                                input_right_slice = all_time_actions_right[:, start_col:end_col, :]
                                left_mask = ~torch.isnan(input_left_slice)
                                right_mask = ~torch.isnan(input_right_slice)
                                combined_mask = torch.cat([left_mask, right_mask], dim=2)
                                
                                input_left_safe = torch.nan_to_num(input_left_slice, nan=0.0)
                                input_right_safe = torch.nan_to_num(input_right_slice, nan=0.0)
                                denorm_left = input_left_safe * stats_left_torch['action_std'] + stats_left_torch['action_mean']
                                denorm_right = input_right_safe * stats_right_torch['action_std'] + stats_right_torch['action_mean']
                                
                                denorm_dual = torch.cat([denorm_left, denorm_right], dim=2)
                                renorm_dual = (denorm_dual - stats_dual_torch['action_mean']) / stats_dual_torch['action_std']
                                
                                val_inherit_dual = renorm_dual.clone()
                                val_inherit_dual[~combined_mask] = float('nan')
                                all_time_actions_dual[:, start_col:end_col, :].copy_(val_inherit_dual)

                    else:
                        print(f"[Step {t}] All commands finished.")
                        break

                # -------------------------------
                # Render & Obs
                # -------------------------------
                # Use observation image (already rendered) instead of re-rendering
                # This saves time. Resolution is typically 320x240.
                curr_viz_image = ts.observation['images']['top']
                
                if onscreen_render:
                    plt_img.set_data(curr_viz_image)
                    plt.pause(DT)

                if args.save_video:
                     # User requested 480p (640x480) for video, but obs is 320x240.
                     # We must re-render for high quality video.
                     video_frame_highres = env._physics.render(height=480, width=640, camera_id='top')
                     video_frames.append(video_frame_highres) 
                
                obs = ts.observation
                qpos_numpy = np.array(obs['qpos'])

                # -------------------------------
                # Policy Query
                # -------------------------------
                with torch.inference_mode():
                    should_plan = False
                    if temporal_agg: should_plan = True
                    elif step_in_chunk == 0 or step_in_chunk >= chunk_size: should_plan = True

                    if should_plan:
                        if not temporal_agg: step_in_chunk = 0

                        # UNIFIED DUAL POLICY logic (for both Coop and Independent-Dual)
                        if current_mode == MODE_COOP or (current_mode == MODE_INDEPENDENT and policy_independent_dual):
                            curr_image = get_image_dual(ts, camera_names)
                            
                            if current_mode == MODE_COOP:
                                qpos = pre_process_dual(qpos_numpy)
                                qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                                action_chunk = policy_dual(qpos, curr_image)
                            else:
                                # Independent (Dual Policy)
                                qpos = pre_process_independent_dual(qpos_numpy)
                                qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                                action_chunk = policy_independent_dual(qpos, curr_image)

                            if temporal_agg:
                                # SHARE THE SAME BUFFER for smooth transition
                                all_time_actions_dual[[t], t:t+num_queries] = action_chunk
                            else:
                                current_action_chunk_dual = action_chunk.squeeze(0).cpu().numpy()
                        else:
                            # Independent (Legacy Split Policy)
                             # Left
                             qpos_left_numpy = qpos_numpy[:7]
                             qpos_left = pre_process_left(qpos_left_numpy)
                             qpos_left = torch.from_numpy(qpos_left).float().cuda().unsqueeze(0)
                             curr_image_left = get_image_independent(ts, camera_names, 'left')
                             
                             action_chunk_l = policy_left(qpos_left, curr_image_left)
                             if temporal_agg:
                                 all_time_actions_left[[t], t:t+num_queries] = action_chunk_l
                             else:
                                 current_action_chunk_left = action_chunk_l.squeeze(0).cpu().numpy()
                            
                            # Right
                             qpos_right_numpy = qpos_numpy[7:14]
                             qpos_right = pre_process_right(qpos_right_numpy)
                             qpos_right = torch.from_numpy(qpos_right).float().cuda().unsqueeze(0)
                             curr_image_right = get_image_independent(ts, camera_names, 'right')
                             
                             action_chunk_r = policy_right(qpos_right, curr_image_right)
                             if temporal_agg:
                                 all_time_actions_right[[t], t:t+num_queries] = action_chunk_r
                             else:
                                 current_action_chunk_right = action_chunk_r.squeeze(0).cpu().numpy()

                    # -------------------------------
                    # Action Combine
                    # -------------------------------
                    if current_mode == MODE_COOP or (current_mode == MODE_INDEPENDENT and policy_independent_dual):
                        if temporal_agg:
                            actions_for_curr_step = all_time_actions_dual[:, t]
                            actions_populated = torch.all(~torch.isnan(actions_for_curr_step), axis=1)
                            actions_for_curr_step = actions_for_curr_step[actions_populated]
                            k = 0.01
                            weights_len = len(actions_for_curr_step)
                            exp_weights = np.exp(-k * (weights_len - 1 - np.arange(weights_len)))
                            exp_weights = exp_weights / exp_weights.sum()
                            exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                            raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                            raw_action = raw_action.squeeze(0).cpu().numpy()
                        else:
                            raw_action = current_action_chunk_dual[step_in_chunk]
                        
                        if current_mode == MODE_COOP:
                             target_qpos = post_process_dual(raw_action)
                        else:
                             target_qpos = post_process_independent_dual(raw_action)
                        
                    else:
                        # Independent (Legacy)
                        offset_t = t # No phase2 offset for general eval
                        
                        if temporal_agg:
                            # LEFT
                            actions_l = all_time_actions_left[:, offset_t]
                            actions_populated_l = torch.all(~torch.isnan(actions_l), axis=1)
                            actions_l = actions_l[actions_populated_l]
                            if len(actions_l) == 0: # Safety fallback
                                 actions_l = all_time_actions_left[:, t]
                                 actions_populated_l = torch.all(~torch.isnan(actions_l), axis=1)
                                 actions_l = actions_l[actions_populated_l]

                            k = 0.01
                            weights_len_l = len(actions_l)
                            exp_weights_l = np.exp(-k * (weights_len_l - 1 - np.arange(weights_len_l)))
                            exp_weights_l = exp_weights_l / exp_weights_l.sum()
                            exp_weights_l = torch.from_numpy(exp_weights_l).cuda().unsqueeze(dim=1)
                            raw_action_l = (actions_l * exp_weights_l).sum(dim=0, keepdim=True)
                            raw_action_l = raw_action_l.squeeze(0).cpu().numpy()

                            # RIGHT
                            actions_r = all_time_actions_right[:, offset_t]
                            actions_populated_r = torch.all(~torch.isnan(actions_r), axis=1)
                            actions_r = actions_r[actions_populated_r]
                            if len(actions_r) == 0:
                                 actions_r = all_time_actions_right[:, t]
                                 actions_populated_r = torch.all(~torch.isnan(actions_r), axis=1)
                                 actions_r = actions_r[actions_populated_r]

                            weights_len_r = len(actions_r)
                            exp_weights_r = np.exp(-k * (weights_len_r - 1 - np.arange(weights_len_r)))
                            exp_weights_r = exp_weights_r / exp_weights_r.sum()
                            exp_weights_r = torch.from_numpy(exp_weights_r).cuda().unsqueeze(dim=1)
                            raw_action_r = (actions_r * exp_weights_r).sum(dim=0, keepdim=True)
                            raw_action_r = raw_action_r.squeeze(0).cpu().numpy()
                        else:
                            idx = step_in_chunk
                            if idx >= chunk_size: idx = chunk_size - 1
                            raw_action_l = current_action_chunk_left[idx]
                            raw_action_r = current_action_chunk_right[idx]

                        action_left = post_process_left(raw_action_l)
                        action_right = post_process_right(raw_action_r)
                        target_qpos = np.concatenate([action_left, action_right])

                ts = env.step(target_qpos)
                current_reward = env._task.get_reward(env.physics)
                
                # Track touched cubes
                new_touches = get_touched_cubes(env.physics)
                current_touched_cubes.update(new_touches)
                
                step_in_chunk += 1
                t += 1
                
                # Safety break
                if t >= MAX_BUFFER_STEPS:
                    print("Max steps reached. Terminating.")
                    break
            
            # --- Episode Finished ---
            total_rewards.append(current_reward)
            is_success = (current_reward >= max_possible_reward)
            if is_success:
                success_count += 1
            
            print(f"Episode {episode_count} Finished. Reward: {current_reward}/{max_possible_reward}. Success: {is_success}")
            
            # Save Video
            if args.save_video and len(video_frames) > 0:
                 status_str = "success" if is_success else "fail"
                 video_path = f'eval_ep{episode_count}_{status_str}_r{current_reward}.mp4'
                 
                 # Detect shape from first frame
                 h, w, _ = video_frames[0].shape
                 fps = 30
                 out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                 for frame in video_frames:
                     frame_bgr = frame[:, :, [2, 1, 0]]
                     out.write(frame_bgr)
                 out.release()
                 print(f"Saved video to {video_path}")
            
            episode_count += 1

        # --- Final Statistics ---
        print("\n" + "="*30)
        print("EVALUATION RESULTS")
        print("="*30)
        print(f"Total Episodes: {episode_count}")
        print(f"Success Rate:   {success_count/episode_count*100:.1f}% ({success_count}/{episode_count})")
        print(f"Average Reward: {np.mean(total_rewards):.2f}")
        print("="*30)

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', action='store', type=str, default='sim_many_cubes')
    parser.add_argument('--ckpt_dual', action='store', type=str, required=False)
    parser.add_argument('--ckpt_left', action='store', type=str, required=False)
    parser.add_argument('--ckpt_right', action='store', type=str, required=False)
    parser.add_argument('--ckpt_independent_dual', action='store', type=str, required=False, help='Unified Dual policy for Independent tasks')
    parser.add_argument('--ckpt_e2e', action='store', type=str, required=False, help='Single E2E policy for ALL tasks (replaces others)')
    
    parser.add_argument('--commands', action='store', type=str, help='Command sequence (e.g. ICI)', required=True)
    parser.add_argument('--color_sequence', action='store', type=str, help='Color sequence', default=None)
    parser.add_argument('--step_i', action='store', type=int, default=400, help='Steps for Independent task')
    parser.add_argument('--step_c', action='store', type=int, default=520, help='Steps for Cooperative task')
    
    parser.add_argument('--policy_class', action='store', type=str, default='ACT')
    parser.add_argument('--kl_weight', action='store', type=int, default=10)
    parser.add_argument('--chunk_size', action='store', type=int, default=100)
    parser.add_argument('--hidden_dim', action='store', type=int, default=512)
    parser.add_argument('--dim_feedforward', action='store', type=int, default=3200)
    parser.add_argument('--lr', action='store', type=float, default=1e-5)
    
    parser.add_argument('--no_temporal_agg', action='store_true')
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--inherit_temporal_buffer', action='store_true', help='Inherit temporal aggregation buffer on switch')
    parser.add_argument('--interleave_objects', action='store_true', help='Interleave last 4 objects among first 5 (High Difficulty)')
    parser.add_argument('--save_video', action='store_true', help='Save execution video')
    parser.add_argument('--num_rollouts', action='store', type=int, default=1, help='Number of evaluation episodes')
    
    args = parser.parse_args()
    main(args)
