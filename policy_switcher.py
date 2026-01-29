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
from piper_sim_env import make_sim_env
from piper_ee_sim_env import make_ee_sim_env
from utils import apply_rgb_mask_to_strip
# Constants
MODE_INDEPENDENT = '1'
MODE_COOP = '2'

def is_data():
    return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])

def get_key():
    if is_data():
        return sys.stdin.read(1)
    return None

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
        
        if arm == 'left':
            curr_image = curr_image[:, :w//2, :]
        else:
            # Shift left by 40 pixels
            offset = 40
            start = w//2 - offset
            end = w - offset
            curr_image = curr_image[:, start:end, :]
            
            # Apply RGB mask
            # curr_image is (H, W, C) which is what the function expects
            curr_image = apply_rgb_mask_to_strip(curr_image, strip_width=offset)
            
        curr_image = rearrange(curr_image, 'h w c -> c h w')
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_image

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
    
    # Handle chunk_size mismatch (e.g. 100 -> 50)
    if 'model.pos_table' in loaded_state_dict and 'model.query_embed.weight' in loaded_state_dict:
        # Check pos_table
        curr_pos_len = policy.model.pos_table.shape[1]
        load_pos_len = loaded_state_dict['model.pos_table'].shape[1]
        if load_pos_len > curr_pos_len:
             # print(f"Trimming pos_table: {load_pos_len} -> {curr_pos_len}")
             loaded_state_dict['model.pos_table'] = loaded_state_dict['model.pos_table'][:, :curr_pos_len, :]
             
        # Check query_embed
        curr_query_len = policy.model.query_embed.weight.shape[0]
        load_query_len = loaded_state_dict['model.query_embed.weight'].shape[0]
        if load_query_len > curr_query_len:
             # print(f"Trimming query_embed: {load_query_len} -> {curr_query_len}")
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

    from piper_constants import SIM_TASK_CONFIGS
    task_config = SIM_TASK_CONFIGS[task_name]
    episode_len = task_config['episode_len']
    camera_names = task_config['camera_names']

    print("Loading Dual Policy...")
    policy_dual, stats_dual = load_policy_and_stats(ckpt_dual, policy_class, args, override_state_dim=14)
    
    print("Loading Left Policy...")
    policy_left, stats_left = load_policy_and_stats(ckpt_left, policy_class, args, override_state_dim=7, override_arm='left')
    
    print("Loading Right Policy...")
    policy_right, stats_right = load_policy_and_stats(ckpt_right, policy_class, args, override_state_dim=7, override_arm='right')

    pre_process_dual = lambda s_qpos: (s_qpos - stats_dual['qpos_mean']) / stats_dual['qpos_std']
    post_process_dual = lambda a: a * stats_dual['action_std'] + stats_dual['action_mean']
    
    pre_process_left = lambda s_qpos: (s_qpos - stats_left['qpos_mean']) / stats_left['qpos_std']
    post_process_left = lambda a: a * stats_left['action_std'] + stats_left['action_mean']
    
    pre_process_right = lambda s_qpos: (s_qpos - stats_right['qpos_mean']) / stats_right['qpos_std']
    post_process_right = lambda a: a * stats_right['action_std'] + stats_right['action_mean']

    # Convert stats to torch for buffer conversion
    def to_torch(x): return torch.from_numpy(x).float().cuda()
    stats_dual_torch = {k: to_torch(v) for k, v in stats_dual.items()}
    stats_left_torch = {k: to_torch(v) for k, v in stats_left.items()}
    stats_right_torch = {k: to_torch(v) for k, v in stats_right.items()}

    # Check for stats mismatch
    print("\n--- Stats Check ---")
    print(f"Dual Mean norm: {torch.norm(stats_dual_torch['action_mean'])}")
    print(f"Left Mean norm: {torch.norm(stats_left_torch['action_mean'])}")
    print(f"Diff Dual[:7] vs Left: {torch.norm(stats_dual_torch['action_mean'][:7] - stats_left_torch['action_mean'])}")
    print("-------------------\n")

    env = make_sim_env(task_name)
    
    if onscreen_render:
        plt.ion()
        ax = plt.subplot()
        plt_img = ax.imshow(env._physics.render(height=480, width=640, camera_id='top'))
    
    def reset_with_new_pose():
        # Pose sampling based on task name
        if 'sim_transfer_cube' in task_name:
            REDBOX_POSE[0] = sample_redbox_pose()
        elif 'sim_moving_cube' in task_name:
            REDBOX_POSE[0] = sample_redbox_pose()
            GREENBOX_POSE[0] = sample_greenbox_pose()
            BLUEBOX_POSE[0] = sample_bluebox_pose()
        elif 'sim_coop' in task_name:
            REDBOX_POSE[0] = sample_redbox_pose()
            GREENBOX_POSE[0] = sample_greenbox_pose()
            BLUEBOX_POSE[0] = sample_bluebox_pose()
        elif 'sim_many_cubes' in task_name:
            # ManyCubesTask randomizes internally, so we don't need to set global poses here.
            pass
        return env.reset()

    ts = reset_with_new_pose()
    
    old_settings = None
    try:
        old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
    except:
        print("Not a TTY, manual control disabled")
    
    # Default to Independent logic or Dual logic depending on request?
    # User changed it to MODE_COOP in Step 162. Keeping MODE_COOP.
    current_mode = MODE_COOP
    
    print("\n\nReady!")
    print("Press '1' for Independent Mode")
    print("Press '2' for Cooperative Mode")
    print("Press 'q' to quit")
    
    chunk_size = args.chunk_size
    # Use episode_len directly as per imitate_episodes (eval_bc)
    max_timesteps = episode_len
    
    # State tracking for chunk execution
    step_in_chunk = 0
    current_action_chunk_dual = None
    current_action_chunk_left = None
    current_action_chunk_right = None
    
    # Initialize Temporal Aggregation Logic
    temporal_agg = not args.no_temporal_agg
    if temporal_agg:
        num_queries = args.chunk_size
        float_nan = float('nan')
        all_time_actions_dual = torch.full([max_timesteps, max_timesteps+num_queries, 14], float_nan).cuda()
        all_time_actions_left = torch.full([max_timesteps, max_timesteps+num_queries, 7], float_nan).cuda()
        all_time_actions_right = torch.full([max_timesteps, max_timesteps+num_queries, 7], float_nan).cuda()


    try:
        t = 0
        video_frames = []
        episode_count = 0
        
        # Stats
        episode_returns = []
        highest_rewards = []
        
        current_episode_rewards = []
        
        while True:
            # Auto-switch logic
            if args.switch_step is not None and t == args.switch_step:
                if current_mode != MODE_INDEPENDENT:
                    current_mode = MODE_INDEPENDENT
                    step_in_chunk = 0 # Force replan on switch
                    print(f"[Step {t}] Auto-switched to INDEPENDENT mode (switch_step={args.switch_step})")
                    
                    if args.inherit_temporal_buffer and temporal_agg:
                        print(f"[Step {t}] Inheriting temporal buffer (Dual -> Indep) with BLENDING")
                        # Un-normalize Dual
                        # Create mask for valid (non-NaN) entries
                        input_actions = all_time_actions_dual
                        dual_mask = ~torch.isnan(input_actions)
                        
                        # Temporarily replace NaNs with 0 to safely calculate denorm/renorm
                        input_actions_safe = torch.nan_to_num(input_actions, nan=0.0)
                        denorm_actions = input_actions_safe * stats_dual_torch['action_std'] + stats_dual_torch['action_mean']
                        
                        # Re-normalize for Left
                        denorm_left = denorm_actions[:, :, :7]
                        renorm_left = (denorm_left - stats_left_torch['action_mean']) / stats_left_torch['action_std']
                        
                        # Apply mask and BLEND if Warmup exists
                        val_inherit_left = renorm_left.clone()
                        val_inherit_left[~dual_mask[:, :, :7]] = float('nan')
                        if args.warmup_steps > 0:
                            # Pre-fill empty slots in new buffer with inherited values
                            # This prevents blending with NaN/0 where new policy has no history
                            empty_mask = torch.isnan(all_time_actions_left)
                            # Only fill valid inherited values
                            valid_inherit_mask = dual_mask[:, :, :7]
                            # Combine: target is empty AND source is valid
                            fill_mask = empty_mask & valid_inherit_mask
                            
                            all_time_actions_left[fill_mask] = val_inherit_left[fill_mask]

                            # Linear Temporal Blending
                            # Create alpha weights for columns [t : t+num_queries]
                            # alpha increases from 0 (Dual) to 1 (Indep)
                            # all_time_actions shape: [max_time, max_time+queries, dim]
                            # We only care about the relevant future window
                            future_len = min(args.chunk_size, all_time_actions_left.shape[1] - t)
                            if future_len > 0:
                                alphas = torch.linspace(0, 1, steps=future_len).float().cuda()
                                alphas = alphas.view(1, -1, 1) # Broadcast: [1, future_len, 1]
                                
                                # Apply to the slice [:, t:t+future_len, :]
                                # Note: val_inherit_left is full size. We need to slice it or apply alpha broadly.
                                # Actually val_inherit_left matches all_time_actions shape.
                                
                                # Create a full-size alpha mask
                                full_alphas = torch.ones_like(all_time_actions_left) # Default 1 (Indep)
                                # Set transition window to 0->1
                                # But we can't easily set diagonal stripes.
                                # Wait, the buffer is [Time_Query_Generated, Time_Action_Executed].
                                # Using columns (Time_Action_Executed) is correct for output smoothness.
                                
                                # Let's overwrite column by column for the transition window
                                for k in range(future_len):
                                    col_idx = t + k
                                    if col_idx < all_time_actions_left.shape[1]:
                                        alpha = float(k) / float(future_len)
                                        # Blend column
                                        all_time_actions_left[:, col_idx, :] = \
                                            all_time_actions_left[:, col_idx, :] * alpha + \
                                            val_inherit_left[:, col_idx, :] * (1 - alpha)
                                            
                                # For columns beyond window (k >= future_len), alpha is implicit 1 (keep Warmup)
                                # For columns before t, we don't care (past).
                        else:
                            all_time_actions_left.copy_(val_inherit_left)
                        
                        # Re-normalize for Right
                        denorm_right = denorm_actions[:, :, 7:]
                        renorm_right = (denorm_right - stats_right_torch['action_mean']) / stats_right_torch['action_std']
                        
                        val_inherit_right = renorm_right.clone()
                        val_inherit_right[~dual_mask[:, :, 7:]] = float('nan')
                        if args.warmup_steps > 0:
                            # Pre-fill empty slots in new buffer with inherited values
                            empty_mask = torch.isnan(all_time_actions_right)
                            valid_inherit_mask = dual_mask[:, :, 7:]
                            fill_mask = empty_mask & valid_inherit_mask
                            
                            all_time_actions_right[fill_mask] = val_inherit_right[fill_mask]

                            future_len = min(args.chunk_size, all_time_actions_right.shape[1] - t)
                            for k in range(future_len):
                                col_idx = t + k
                                if col_idx < all_time_actions_right.shape[1]:
                                    alpha = float(k) / float(future_len)
                                    all_time_actions_right[:, col_idx, :] = \
                                        all_time_actions_right[:, col_idx, :] * alpha + \
                                        val_inherit_right[:, col_idx, :] * (1 - alpha)
                        else:
                            all_time_actions_right.copy_(val_inherit_right)
                    
                    if args.warmup_steps > 0:
                         print(f"[Step {t}] Switched to INDEPENDENT mode (Warmed up for {args.warmup_steps} steps)")

            key = None
            if old_settings:
                key = get_key()
            if key == '1':
                if current_mode != MODE_INDEPENDENT:
                    current_mode = MODE_INDEPENDENT
                    step_in_chunk = 0 # Force replan on switch
                    print(f"[Step {t}] Switched to INDEPENDENT mode")

                    if args.inherit_temporal_buffer and temporal_agg:
                        print(f"[Step {t}] Inheriting temporal buffer (Dual -> Indep)")
                        # Un-normalize Dual
                        # Create mask for valid (non-NaN) entries
                        input_actions = all_time_actions_dual
                        dual_mask = ~torch.isnan(input_actions)
                        
                        input_actions_safe = torch.nan_to_num(input_actions, nan=0.0)
                        denorm_actions = input_actions_safe * stats_dual_torch['action_std'] + stats_dual_torch['action_mean']

                        # Re-normalize Left/Right
                        denorm_left = denorm_actions[:, :, :7]
                        renorm_left = (denorm_left - stats_left_torch['action_mean']) / stats_left_torch['action_std']
                        
                        val_inherit_left = renorm_left.clone()
                        val_inherit_left[~dual_mask[:, :, :7]] = float('nan')
                        all_time_actions_left.copy_(val_inherit_left)
                        
                        denorm_right = denorm_actions[:, :, 7:]
                        renorm_right = (denorm_right - stats_right_torch['action_mean']) / stats_right_torch['action_std']
                        
                        val_inherit_right = renorm_right.clone()
                        val_inherit_right[~dual_mask[:, :, 7:]] = float('nan')
                        all_time_actions_right.copy_(val_inherit_right)
            elif key == '2':
                if current_mode != MODE_COOP:
                    current_mode = MODE_COOP
                    step_in_chunk = 0 # Force replan on switch
                    print(f"[Step {t}] Switched to COOPERATIVE mode")

                    if args.inherit_temporal_buffer and temporal_agg:
                        print(f"[Step {t}] Inheriting temporal buffer (Indep -> Dual)")
                        
                        input_left = all_time_actions_left
                        input_right = all_time_actions_right
                        
                        # Masks
                        left_mask = ~torch.isnan(input_left)
                        right_mask = ~torch.isnan(input_right)
                        combined_mask = torch.cat([left_mask, right_mask], dim=2)
                        
                        input_left_safe = torch.nan_to_num(input_left, nan=0.0)
                        input_right_safe = torch.nan_to_num(input_right, nan=0.0)
                        
                        denorm_left = input_left_safe * stats_left_torch['action_std'] + stats_left_torch['action_mean']
                        denorm_right = input_right_safe * stats_right_torch['action_std'] + stats_right_torch['action_mean']

                        # Re-normalize for Dual
                        renorm_dual = (denorm_dual - stats_dual_torch['action_mean']) / stats_dual_torch['action_std']
                        
                        val_inherit_dual = renorm_dual.clone()
                        val_inherit_dual[~combined_mask] = float('nan')
                        all_time_actions_dual.copy_(val_inherit_dual)
            elif key == 'q':
                break
                
            # Render update matching imitate_episodes.py timing
            if onscreen_render:
                image = env._physics.render(height=480, width=640, camera_id='top')
                plt_img.set_data(image)
                plt.pause(DT)

            if args.save_video:
                 # Capture for video even if onscreen_render is False? 
                 # Usually users want both or just video. 
                 # Let's reuse 'image' if available, else render.
                 if not onscreen_render:
                      image = env._physics.render(height=480, width=640, camera_id='top')
                 video_frames.append(image) 
            
            obs = ts.observation
            qpos_numpy = np.array(obs['qpos'])
            
            with torch.inference_mode():
                # If we need to plan
                should_plan = False
                if temporal_agg:
                    should_plan = True
                elif step_in_chunk == 0 or step_in_chunk >= chunk_size:
                    should_plan = True

                if should_plan:
                    if not temporal_agg:
                        step_in_chunk = 0 # Reset only if not agg

                    if current_mode == MODE_COOP:
                        qpos = pre_process_dual(qpos_numpy)
                        qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                        curr_image = get_image_dual(ts, camera_names)
                        
                        action_chunk = policy_dual(qpos, curr_image) # [1, chunk_size, 14]
                        if temporal_agg:
                            all_time_actions_dual[[t], t:t+num_queries] = action_chunk
                        else:
                            current_action_chunk_dual = action_chunk.squeeze(0).cpu().numpy()

                        # Warmup Logic (Shadow Mode)
                        if args.switch_step is not None and args.warmup_steps > 0:
                            steps_until_switch = args.switch_step - t
                            if steps_until_switch > 0 and steps_until_switch <= args.warmup_steps:
                                if steps_until_switch % 10 == 0: # minimal log
                                    print(f"[Step {t}] Warming up Independent policies...")
                                # Query Left
                                qpos_left_numpy = qpos_numpy[:7]
                                qpos_left = pre_process_left(qpos_left_numpy)
                                qpos_left = torch.from_numpy(qpos_left).float().cuda().unsqueeze(0)
                                curr_image_left = get_image_independent(ts, camera_names, 'left')
                                action_chunk_l = policy_left(qpos_left, curr_image_left)
                                all_time_actions_left[[t], t:t+num_queries] = action_chunk_l

                                # Query Right
                                qpos_right_numpy = qpos_numpy[7:14]
                                qpos_right = pre_process_right(qpos_right_numpy)
                                qpos_right = torch.from_numpy(qpos_right).float().cuda().unsqueeze(0)
                                curr_image_right = get_image_independent(ts, camera_names, 'right')
                                action_chunk_r = policy_right(qpos_right, curr_image_right)
                                all_time_actions_right[[t], t:t+num_queries] = action_chunk_r

                    else:
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
                         
                         if args.policy_class == 'ACT':
                             action_chunk_r = policy_right(qpos_right, curr_image_right)
                             if temporal_agg:
                                 all_time_actions_right[[t], t:t+num_queries] = action_chunk_r
                             else:
                                 current_action_chunk_right = action_chunk_r.squeeze(0).cpu().numpy()
                             
                
                # Execute current step of the plan
                if current_mode == MODE_COOP:
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
                    
                    action = post_process_dual(raw_action)
                    target_qpos = action
                else:
                    # Independent Mode
                    offset_t = t
                    if args.phase2_offset > 0:
                        offset_t = t + args.phase2_offset

                    if temporal_agg:
                        # LEFT
                        actions_for_curr_step_l = all_time_actions_left[:, offset_t]
                        actions_populated_l = torch.all(~torch.isnan(actions_for_curr_step_l), axis=1)
                        actions_for_curr_step_l = actions_for_curr_step_l[actions_populated_l]
                        
                        # Handle case where offset pushes into unpopulated territory (though usually fine with ACT)
                        if len(actions_for_curr_step_l) == 0:
                             # Fallback to current t if offset is too far (safety)
                             actions_for_curr_step_l = all_time_actions_left[:, t]
                             actions_populated_l = torch.all(~torch.isnan(actions_for_curr_step_l), axis=1)
                             actions_for_curr_step_l = actions_for_curr_step_l[actions_populated_l]

                        k = 0.01
                        weights_len_l = len(actions_for_curr_step_l)
                        exp_weights_l = np.exp(-k * (weights_len_l - 1 - np.arange(weights_len_l)))
                        exp_weights_l = exp_weights_l / exp_weights_l.sum()
                        exp_weights_l = torch.from_numpy(exp_weights_l).cuda().unsqueeze(dim=1)
                        raw_action_l = (actions_for_curr_step_l * exp_weights_l).sum(dim=0, keepdim=True)
                        raw_action_l = raw_action_l.squeeze(0).cpu().numpy()

                        # RIGHT
                        actions_for_curr_step_r = all_time_actions_right[:, offset_t]
                        actions_populated_r = torch.all(~torch.isnan(actions_for_curr_step_r), axis=1)
                        actions_for_curr_step_r = actions_for_curr_step_r[actions_populated_r]
                        
                        if len(actions_for_curr_step_r) == 0:
                             actions_for_curr_step_r = all_time_actions_right[:, t]
                             actions_populated_r = torch.all(~torch.isnan(actions_for_curr_step_r), axis=1)
                             actions_for_curr_step_r = actions_for_curr_step_r[actions_populated_r]

                        k = 0.01
                        weights_len_r = len(actions_for_curr_step_r)
                        exp_weights_r = np.exp(-k * (weights_len_r - 1 - np.arange(weights_len_r)))
                        exp_weights_r = exp_weights_r / exp_weights_r.sum()
                        exp_weights_r = torch.from_numpy(exp_weights_r).cuda().unsqueeze(dim=1)
                        raw_action_r = (actions_for_curr_step_r * exp_weights_r).sum(dim=0, keepdim=True)
                        raw_action_r = raw_action_r.squeeze(0).cpu().numpy()
                    else:
                        idx = step_in_chunk + args.phase2_offset
                        if idx >= chunk_size:
                             idx = chunk_size - 1 # Clamp
                        raw_action_l = current_action_chunk_left[idx]
                        raw_action_r = current_action_chunk_right[idx]
                        
                    action_left = post_process_left(raw_action_l)
                    action_right = post_process_right(raw_action_r)
                    target_qpos = np.concatenate([action_left, action_right])
            
            ts = env.step(target_qpos)
            current_episode_rewards.append(ts.reward)
            
            step_in_chunk += 1
            t += 1
            
            # Reset logic matches imitate_episodes num_rollouts loop (conceptually)
            if t >= max_timesteps:
                print("Episode finished. Resetting...")
                
                if args.save_video and len(video_frames) > 0:
                     video_path = f'sim_policy_switch_ep{episode_count}.mp4'
                     h, w, _ = video_frames[0].shape
                     fps = 30
                     out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                     for frame in video_frames:
                         # Mujoco returns RGB, OpenCV needs BGR
                         frame_bgr = frame[:, :, [2, 1, 0]]
                         out.write(frame_bgr)
                     out.release()
                     print(f"Saved video to {video_path}")
                     video_frames = []


                # Result recording
                rewards = np.array(current_episode_rewards)
                episode_return = np.sum(rewards[rewards!=None])
                episode_returns.append(episode_return)
                episode_highest_reward = np.max(rewards)
                highest_rewards.append(episode_highest_reward)
                print(f"Episode {episode_count}: Return={episode_return}, MaxReward={episode_highest_reward}")
                
                episode_count += 1
                current_episode_rewards = []

                if episode_count >= args.num_rollouts:
                    break

                ts = reset_with_new_pose()
                t = 0
                step_in_chunk = 0
                current_mode = MODE_COOP # Reset mode to Coop for new episode
                
                # Reset temp buffers for next episode
                if temporal_agg:
                    all_time_actions_dual.fill_(float_nan)
                    all_time_actions_left.fill_(float_nan)
                    all_time_actions_right.fill_(float_nan)
                    
        # Summary
        success_rate = np.mean(np.array(highest_rewards) == 4) # Assuming 4 is max reward for coop
        avg_return = np.mean(episode_returns)
        print(f"\nEvaluation Finished.")
        print(f"Success Rate (MaxReward=4): {success_rate*100:.1f}%")
        print(f"Average Return: {avg_return:.2f}")

    except KeyboardInterrupt:
        print("\nStopped.")


    finally:
        if old_settings:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', action='store', type=str, required=True)
    parser.add_argument('--ckpt_dual', action='store', type=str, required=True)
    parser.add_argument('--ckpt_left', action='store', type=str, required=True)
    parser.add_argument('--ckpt_right', action='store', type=str, required=True)
    
    parser.add_argument('--policy_class', action='store', type=str, default='ACT')
    parser.add_argument('--kl_weight', action='store', type=int, default=10)
    parser.add_argument('--chunk_size', action='store', type=int, default=100)
    parser.add_argument('--hidden_dim', action='store', type=int, default=512)
    parser.add_argument('--dim_feedforward', action='store', type=int, default=3200)
    parser.add_argument('--lr', action='store', type=float, default=1e-5)
    
    parser.add_argument('--no_temporal_agg', action='store_true')
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--switch_step', action='store', type=int, help='Step to auto-switch from Coop to Independent')
    parser.add_argument('--warmup_steps', action='store', type=int, default=0, help='Number of steps to run independent policy in background before switch')
    parser.add_argument('--inherit_temporal_buffer', action='store_true', help='Inherit temporal aggregation buffer on switch to prevent jerk')
    parser.add_argument('--save_video', action='store_true', help='Save execution video to mp4')
    parser.add_argument('--num_rollouts', action='store', type=int, default=1, help='Number of evaluation episodes')
    parser.add_argument('--phase2_offset', action='store', type=int, default=0, help='Execute Phase 2 actions N steps earlier')
    
    args = parser.parse_args()
    main(args)
