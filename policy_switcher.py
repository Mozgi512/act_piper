import sys
import os
import tty
import termios
import select
import time
from copy import deepcopy
import pickle
import argparse
import json
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
from piper_constants import DT, START_ARM_POSE
from piper_constants import PUPPET_GRIPPER_JOINT_OPEN
from utils import load_data
from utils import sample_redbox_pose, sample_insertion_pose, sample_greenbox_pose, sample_bluebox_pose
from utils import compute_dict_mean, set_seed, detach_dict
from policy import ACTPolicy

# Import Sim Env
from piper_sim_env import REDBOX_POSE, GREENBOX_POSE, BLUEBOX_POSE, MANYCUBES_COLORS
from piper_sim_env import make_sim_env
from piper_ee_sim_env import make_ee_sim_env
from utils import apply_rgb_mask_to_strip, apply_rgb_mask_to_right_strip
# Constants
MODE_INDEPENDENT = '1'
MODE_COOP = '2'

class TaskScheduler:
    def __init__(self, sequence_str, duration_config):
        self.sequence_str = sequence_str
        self.config = duration_config
        self.mode_schedule = {} # step -> mode
        self.hold_schedule = {'left': [], 'right': []} 
        
        self.timeline_left = [] # List of {'start':, 'end':, 'type':, 'info':}
        self.timeline_right = []
        
        self.max_timesteps = 0
        
        self.MODE_INDEPENDENT = MODE_INDEPENDENT
        self.MODE_COOP = MODE_COOP
        
        self._calculate_schedule()

    def _calculate_schedule(self):
        time_l = 0
        time_r = 0
        
        n = len(self.sequence_str)
        for i in range(n):
            char = self.sequence_str[i]
            
            if char == 'I':
                dur = self.config.get('I', 0)
                if dur == 0: print(f"WARNING: Duration for 'I' is 0!")
                
                # Independent Parallel
                start_l = time_l
                end_l = start_l + dur
                self.timeline_left.append({'start': start_l, 'end': end_l, 'type': 'INDEP', 'info': 'Parallel I'})
                time_l = end_l
                
                start_r = time_r
                end_r = start_r + dur
                self.timeline_right.append({'start': start_r, 'end': end_r, 'type': 'INDEP', 'info': 'Parallel I'})
                time_r = end_r
                
            elif char == 'L':
                dur = self.config.get('Single', 0)
                # Left Only
                start_l = time_l
                end_l = start_l + dur
                self.timeline_left.append({'start': start_l, 'end': end_l, 'type': 'INDEP', 'info': 'Single L'})
                time_l = end_l
                
            elif char == 'R':
                dur = self.config.get('Single', 0)
                # Right Only
                start_r = time_r
                end_r = start_r + dur
                self.timeline_right.append({'start': start_r, 'end': end_r, 'type': 'INDEP', 'info': 'Single R'})
                time_r = end_r

            elif char == 'C':
                len_assembly = self.config.get('C_assembly', 0)
                len_place = self.config.get('C_place', 0)
                
                # Sync Point
                start_coop = max(time_l, time_r)
                
                # Schedule Holds
                if time_l < start_coop:
                    self.timeline_left.append({'start': time_l, 'end': start_coop, 'type': 'HOLD', 'info': 'Sync Wait'})
                    time_l = start_coop
                
                if time_r < start_coop:
                    self.timeline_right.append({'start': time_r, 'end': start_coop, 'type': 'HOLD', 'info': 'Sync Wait'})
                    time_r = start_coop
                
                # Mode Switch Registration
                self.mode_schedule[start_coop] = self.MODE_COOP
                
                # Phase 1: Assembly (Coop Mode)
                end_assembly = start_coop + len_assembly
                self.timeline_left.append({'start': start_coop, 'end': end_assembly, 'type': 'COOP', 'info': 'Phase 1 (Assembly)'})
                self.timeline_right.append({'start': start_coop, 'end': end_assembly, 'type': 'COOP', 'info': 'Phase 1 (Assembly)'})
                
                # Phase 2: Placement (Indep Mode)
                self.mode_schedule[end_assembly] = self.MODE_INDEPENDENT
                
                # Lookahead for Role Assignment
                base_arm = 'right' # Default
                if i + 1 < n:
                    next_char = self.sequence_str[i+1]
                    if next_char == 'L':
                        base_arm = 'right'
                    elif next_char == 'R':
                        base_arm = 'left'
                    else:
                        base_arm = 'right'
                
                end_place = end_assembly + len_place
                
                if base_arm == 'left':
                    # Left blocked (Base), Right free (Top)
                    self.timeline_left.append({'start': end_assembly, 'end': end_place, 'type': 'INDEP', 'info': 'Phase 2 (Place-Base)'})
                    time_l = end_place
                    time_r = end_assembly # Right free immediately
                else:
                    # Right blocked (Base), Left free (Top)
                    self.timeline_right.append({'start': end_assembly, 'end': end_place, 'type': 'INDEP', 'info': 'Phase 2 (Place-Base)'})
                    time_r = end_place
                    time_l = end_assembly # Left free immediately
                
        self.max_timesteps = max(time_l, time_r)
        
    def print_schedule(self):
        print("\n=== Left Arm Schedule ===")
        print(f"{'Start':<6} | {'End':<6} | {'Type':<10} | {'Info'}")
        for item in self.timeline_left:
            print(f"{item['start']:<6} | {item['end']:<6} | {item['type']:<10} | {item['info']}")
        print("\n=== Right Arm Schedule ===")
        print(f"{'Start':<6} | {'End':<6} | {'Type':<10} | {'Info'}")
        for item in self.timeline_right:
            print(f"{item['start']:<6} | {item['end']:<6} | {item['type']:<10} | {item['info']}")
        print("=======================\n")

    def get_mode_at_step(self, t):
        current = self.MODE_INDEPENDENT
        sorted_steps = sorted(self.mode_schedule.keys())
        for step in sorted_steps:
            if step <= t:
                current = self.mode_schedule[step]
            else:
                break
        return current

    def get_arm_state(self, t, arm):
        timeline = self.timeline_left if arm == 'left' else self.timeline_right
        
        for item in timeline:
            if item['start'] <= t < item['end']:
                return item['type'], item['info']
        
        return 'HOLD', 'Idle/Finished'
    
    def should_hold(self, t, arm):
        state, info = self.get_arm_state(t, arm)
        return state == 'HOLD'

    def get_task_ending_at(self, t, arm):
        """Return the task dict that ends EXACTLY at t."""
        timeline = self.timeline_left if arm == 'left' else self.timeline_right
        for item in timeline:
            if item['end'] == t:
                return item
        return None

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

def apply_torch_rgb_mask(images, strip_width=40, arm='right'):
    """
    Applies RGB mask to a strip of the images (Batch, Cam, C, H, W) or (C, H, W).
    Preserves R, G, B colors; blacks out everything else in the strip.
    arm='right' -> Mask Left Strip (for Right Arm view).
    arm='left' -> Mask Right Strip (for Left Arm view).
    """
    is_single = len(images.shape) == 3
    if is_single:
        images = images.unsqueeze(0)
    
    # Flatten if (B, Cam, C, H, W)
    orig_shape = images.shape
    if len(orig_shape) == 5:
        b, n_cam, c, h, w = orig_shape
        images = images.view(b * n_cam, c, h, w)
        
    B, C, H, W = images.shape
    
    if arm == 'left':
        # Mask Right Strip
        strip_start = W - strip_width
        strip_end = W
    else:
        # Mask Left Strip
        strip_start = 0
        strip_end = strip_width
        
    strip = images[:, :, :, strip_start:strip_end]
    
    # Thresholds (matching independent_imitate_episodes.py)
    t_100 = 100.0 / 255.0
    t_150 = 150.0 / 255.0
    
    r = strip[:, 0, :, :]
    g = strip[:, 1, :, :]
    b = strip[:, 2, :, :]
    
    mask_r = (r > t_100) & (g < t_100) & (b < t_100)
    mask_g = (g > t_100) & (r < t_100) & (b < t_100)
    mask_b = (b > t_150) & (r < t_100) & (g < t_100)
    
    combined_mask = mask_r | mask_g | mask_b
    combined_mask = combined_mask.unsqueeze(1).repeat(1, C, 1, 1).float()
    
    masked_strip = strip * combined_mask
    
    outputs = images.clone()
    outputs[:, :, :, strip_start:strip_end] = masked_strip
    
    if len(orig_shape) == 5:
        outputs = outputs.view(orig_shape)
    elif is_single:
        outputs = outputs.squeeze(0)
        
    return outputs

def remove_cubes(physics, indices):
    """Teleport cubes to z=-10 to effectively remove them."""
    if not indices: return
    for i in indices:
        try:
            start_idx = physics.model.name2id(f'cube_{i}_joint', 'joint')
            qpos_adr = physics.model.jnt_qposadr[start_idx]
            # Set Z to -10, and also move X/Y far away to be safe
            physics.data.qpos[qpos_adr + 0] = 10.0 + i # X far
            physics.data.qpos[qpos_adr + 1] = 10.0 + i # Y far
            physics.data.qpos[qpos_adr + 2] = -10.0    # Z under
            print(f"Removed cube {i}")
        except Exception as e:
            pass

def get_touched_cubes_per_arm(physics):
    """Return dict of sets of cube indices contacted by each gripper."""
    touched = {'left': set(), 'right': set()}
    for i_contact in range(physics.data.ncon):
        id_geom_1 = physics.data.contact[i_contact].geom1
        id_geom_2 = physics.data.contact[i_contact].geom2
        name_1 = physics.model.id2name(id_geom_1, 'geom')
        name_2 = physics.model.id2name(id_geom_2, 'geom')
        
        if name_1 is None or name_2 is None: continue
        
        # Check pair (gripper, cube)
        for n1, n2 in [(name_1, name_2), (name_2, name_1)]:
            # Check for grippers using known keywords in geom names
            # Left: l_gripper_finger, Right: r_gripper_finger
            arm = None
            if 'l_gripper_finger' in n1: arm = 'left'
            elif 'r_gripper_finger' in n1: arm = 'right'
            
            if arm and 'cube_' in n2:
                # Extract index
                try:
                    parts = n2.split('_')
                    # Format usually cube_N or cube_N_something
                    idx_str = parts[1]
                    idx = int(idx_str)
                    touched[arm].add(idx)
                except:
                    pass
    return touched

def get_image_dual(ts, camera_names):
    curr_images = []
    for cam_name in camera_names:
        # Sim Env render (H, W, C)
        curr_image_np = ts.observation['images'][cam_name].copy()
        # Convert to Tensor (1, C, H, W)
        curr_image_t = torch.from_numpy(curr_image_np).permute(2, 0, 1).unsqueeze(0).float().cuda() / 255.0
        curr_images.append(curr_image_t.squeeze(0))
    curr_image = torch.stack(curr_images, dim=0).unsqueeze(0) # (1, num_cam, C, H, W)
    return curr_image

def get_image_independent(ts, camera_names, arm):
    curr_images = []
    for cam_name in camera_names:
        # Sim Env render (H, W, C)
        curr_image_np = ts.observation['images'][cam_name].copy()
        h, w, c = curr_image_np.shape
        
        # Convert to Tensor (1, C, H, W)
        curr_image_t = torch.from_numpy(curr_image_np).permute(2, 0, 1).unsqueeze(0).float().cuda() / 255.0
        
        if arm == 'left':
             # Left Arm: Left Half
             curr_image_t = curr_image_t[:, :, :, :w//2]
             # Mask Right Strip (Overlap with Right)
             curr_image_t = apply_torch_rgb_mask(curr_image_t, strip_width=40, arm='left')
        else:
             # Right Arm: Right Half
             curr_image_t = curr_image_t[:, :, :, w//2:]
             # Mask Left Strip (Overlap with Left)
             curr_image_t = apply_torch_rgb_mask(curr_image_t, strip_width=40, arm='right')
             
        curr_images.append(curr_image_t.squeeze(0))
        
    curr_image = torch.stack(curr_images, dim=0).unsqueeze(0) # (1, num_cam, C, H, W)
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
    
    # Handle Color Sequence
    if args.color_sequence:
        color_seq = list(args.color_sequence.lower())
        if len(color_seq) != 10:
             print(f"WARNING: Color sequence must be 10 chars. Got {len(color_seq)}. Using default.")
        elif any(c not in ['r', 'g', 'b'] for c in color_seq):
             print(f"WARNING: Invalid chars in color sequence. Using default.")
        else:
             print(f"Setting Color Sequence: {color_seq}")
             MANYCUBES_COLORS[0] = color_seq

    print("Loading Left Policy...")
    policy_left, stats_left = load_policy_and_stats(ckpt_left, policy_class, args, override_state_dim=7, override_arm='left')
    
    print("Loading Right Policy...")
    policy_right, stats_right = load_policy_and_stats(ckpt_right, policy_class, args, override_state_dim=7, override_arm='right')
    print(f"Stats Right Action Std Mean: {stats_right['action_std'].mean()}")

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

    chunk_size = args.chunk_size
    # Default duration from task config
    max_timesteps = episode_len
    # Override if requested
    if args.episode_len is not None:
        max_timesteps = args.episode_len
    
    # Initialize Task Scheduler EARLY to determine total time
    scheduler = None
    if args.command_sequence and args.task_durations:
        try:
            durations = json.loads(args.task_durations)
            print(f"Initializing Task Scheduler with Sequence: {args.command_sequence}")
            scheduler = TaskScheduler(args.command_sequence, durations)
            print(f"Total Scheduled Duration: {scheduler.max_timesteps}")
            scheduler.print_schedule()
            
            # Update max_timesteps to scheduler's duration if larger
            if scheduler.max_timesteps > max_timesteps:
                print(f"Updating max_timesteps from {max_timesteps} to {scheduler.max_timesteps} (Scheduler dictated)")
                max_timesteps = scheduler.max_timesteps
        except Exception as e:
            print(f"Error initializing scheduler: {e}")

    # Now create Env with correct time_limit
    time_limit = (max_timesteps + 200) * DT # Add safety buffer steps
    print(f"Creating Simulation Environment with time_limit={time_limit:.2f}s ({max_timesteps} steps + buffer)")
    env = make_sim_env(task_name, time_limit=time_limit)
    
    if onscreen_render:
        plt.ion()
        ax = plt.subplot()
        plt_img = ax.imshow(env._physics.render(height=240, width=320, camera_id='top'))
    
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
    
    # Default to Cooperative logic unless overridden by scheduler later
    current_mode = MODE_COOP
    
    print("\n\nReady!")
    print("Press '1' for Independent Mode")
    print("Press '2' for Cooperative Mode")
    print("Press 'q' to quit")
    
    # State tracking for chunk execution
    step_in_chunk = 0
    current_action_chunk_dual = None
    current_action_chunk_left = None
    current_action_chunk_right = None
    
    # Tracking for status logging
    last_l_state = None
    last_r_state = None
    
    # Tracking for Object Removal
    acc_touched_left = set()
    acc_touched_right = set()
    
    # Initialize current_mode
    if scheduler:
        current_mode = scheduler.get_mode_at_step(0)
        print(f"Initial mode set from scheduler: {current_mode}")
    else:
        current_mode = MODE_COOP
    
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
        

        

        
        # Prepare Home Pose for Holding (14 dim)
        home_pose = np.zeros(14)
        home_pose[:6] = START_ARM_POSE[:6]
        home_pose[6] = START_ARM_POSE[6]
        home_pose[7:13] = START_ARM_POSE[8:14]
        home_pose[13] = START_ARM_POSE[14]

        def handle_mode_switch(t, from_mode, to_mode):
            """Unified mode switch with inheritance and blending."""
            nonlocal current_mode, step_in_chunk
            if from_mode == to_mode: return
            
            print(f"[Step {t}] Switching from {from_mode} to {to_mode}")
            current_mode = to_mode
            step_in_chunk = 0 # Force replan
            
            if not (args.inherit_temporal_buffer and temporal_agg):
                return
                
            if to_mode == MODE_INDEPENDENT:
                print(f"[Step {t}] Inheriting temporal buffer (Dual -> Indep) with BLENDING")
                # Un-normalize Dual
                input_actions = all_time_actions_dual
                dual_mask_val = ~torch.isnan(input_actions)
                input_actions_safe = torch.nan_to_num(input_actions, nan=0.0)
                denorm_actions = input_actions_safe * stats_dual_torch['action_std'] + stats_dual_torch['action_mean']
                
                # Re-normalize for Left
                denorm_left = denorm_actions[:, :, :7]
                renorm_left = (denorm_left - stats_left_torch['action_mean']) / stats_left_torch['action_std']
                val_inherit_left = renorm_left.clone()
                val_inherit_left[~dual_mask_val[:, :, :7]] = float('nan')
                
                if args.warmup_steps > 0:
                    # Fill emtpy slots
                    empty_mask_l = torch.isnan(all_time_actions_left)
                    fill_mask_l = empty_mask_l & dual_mask_val[:, :, :7]
                    all_time_actions_left[fill_mask_l] = val_inherit_left[fill_mask_l]
                    
                    # Blend
                    future_len = min(args.chunk_size, all_time_actions_left.shape[1] - t)
                    for k in range(future_len):
                        col_idx = t + k
                        if col_idx < all_time_actions_left.shape[1]:
                            alpha = float(k) / float(future_len)
                            all_time_actions_left[:, col_idx, :] = \
                                all_time_actions_left[:, col_idx, :] * alpha + \
                                val_inherit_left[:, col_idx, :] * (1 - alpha)
                else:
                    all_time_actions_left.copy_(val_inherit_left)
                
                # Re-normalize for Right
                denorm_right = denorm_actions[:, :, 7:]
                renorm_right = (denorm_right - stats_right_torch['action_mean']) / stats_right_torch['action_std']
                val_inherit_right = renorm_right.clone()
                val_inherit_right[~dual_mask_val[:, :, 7:]] = float('nan')
                
                if args.warmup_steps > 0:
                    empty_mask_r = torch.isnan(all_time_actions_right)
                    fill_mask_r = empty_mask_r & dual_mask_val[:, :, 7:]
                    all_time_actions_right[fill_mask_r] = val_inherit_right[fill_mask_r]
                    
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
                    
            elif to_mode == MODE_COOP:
                print(f"[Step {t}] Inheriting temporal buffer (Indep -> Dual)")
                input_left = all_time_actions_left
                input_right = all_time_actions_right
                
                left_mask_val = ~torch.isnan(input_left)
                right_mask_val = ~torch.isnan(input_right)
                combined_mask_val = torch.cat([left_mask_val, right_mask_val], dim=2)
                
                input_left_safe = torch.nan_to_num(input_left, nan=0.0)
                input_right_safe = torch.nan_to_num(input_right, nan=0.0)
                
                # Fix denorm_dual bug
                denorm_l = input_left_safe * stats_left_torch['action_std'] + stats_left_torch['action_mean']
                denorm_r = input_right_safe * stats_right_torch['action_std'] + stats_right_torch['action_mean']
                denorm_dual_val = torch.cat([denorm_l, denorm_r], dim=2)
                
                renorm_dual_val = (denorm_dual_val - stats_dual_torch['action_mean']) / stats_dual_torch['action_std']
                
                val_inherit_dual = renorm_dual_val.clone()
                val_inherit_dual[~combined_mask_val] = float('nan')
                all_time_actions_dual.copy_(val_inherit_dual)

        while True:
            # Task Scheduler Mode Logic
            if scheduler:
                sched_mode = scheduler.get_mode_at_step(t)
                if sched_mode != current_mode:
                    handle_mode_switch(t, current_mode, sched_mode)
                
                # 2. Status Logging (Per Arm)
                l_state, l_info = scheduler.get_arm_state(t, 'left')
                r_state, r_info = scheduler.get_arm_state(t, 'right')
                
                # Print only on change or periodically
                if (l_state, l_info) != last_l_state or (r_state, r_info) != last_r_state:
                     print(f"[Step {t}] Left: {l_state} ({l_info}) | Right: {r_state} ({r_info})")
                     last_l_state = (l_state, l_info)
                     last_r_state = (r_state, r_info)

            # Auto-switch logic (Legacy)
            elif args.switch_step is not None and t == args.switch_step: # Only if scheduler NOT active
                handle_mode_switch(t, current_mode, MODE_INDEPENDENT)

            key = None
            if old_settings:
                key = get_key()
            if key == '1':
                handle_mode_switch(t, current_mode, MODE_INDEPENDENT)
            elif key == '2':
                handle_mode_switch(t, current_mode, MODE_COOP)
            elif key == 'q':
                break
                
            # Render update matching imitate_episodes.py timing
            if onscreen_render:
                image = env._physics.render(height=240, width=320, camera_id='top')
                plt_img.set_data(image)
                plt.pause(DT)

            if args.save_video:
                 # Capture for video even if onscreen_render is False? 
                 # Usually users want both or just video. 
                 # Let's reuse 'image' if available, else render.
                 if not onscreen_render:
                      image = env._physics.render(height=240, width=320, camera_id='top')
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
            
            # Apply Scheduler Holds
            if scheduler:
                hold_l = scheduler.should_hold(t, 'left')
                hold_r = scheduler.should_hold(t, 'right')
                
                if hold_l:
                    # Override Left Arm to Home
                    target_qpos[:7] = home_pose[:7]
                
                if hold_r:
                    # Override Right Arm to Home
                    target_qpos[7:] = home_pose[7:]





            ts = env.step(target_qpos)
            current_episode_rewards.append(ts.reward)
            
            # --- Object Removal Logic ---
            new_touches = get_touched_cubes_per_arm(env.physics)
            acc_touched_left.update(new_touches['left'])
            acc_touched_right.update(new_touches['right'])
            
            step_in_chunk += 1
            t += 1
            
            # Check for task completion & Remove objects
            if scheduler:
                # Left
                ended_task_l = scheduler.get_task_ending_at(t, 'left')
                if ended_task_l:
                    # Logic: Remove unless it is 'Phase 1' (mid-assembly)
                    if 'Phase 1' not in ended_task_l['info']:
                        print(f"[Step {t}] Left Task Ended ({ended_task_l['info']}). Removing objects: {acc_touched_left}")
                        remove_cubes(env.physics, list(acc_touched_left))
                        acc_touched_left.clear()
                    else:
                        print(f"[Step {t}] Left Phase 1 Ended. Persisting objects {acc_touched_left} for Phase 2.")
                
                # Right
                ended_task_r = scheduler.get_task_ending_at(t, 'right')
                if ended_task_r:
                    if 'Phase 1' not in ended_task_r['info']:
                        print(f"[Step {t}] Right Task Ended ({ended_task_r['info']}). Removing objects: {acc_touched_right}")
                        remove_cubes(env.physics, list(acc_touched_right))
                        acc_touched_right.clear()
                    else:
                        print(f"[Step {t}] Right Phase 1 Ended. Persisting objects {acc_touched_right} for Phase 2.")

            
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
                # Reset mode from scheduler if present
                if scheduler:
                    current_mode = scheduler.get_mode_at_step(0)
                else:
                    current_mode = MODE_COOP
                
                # Reset temp buffers for next episode
                if temporal_agg:
                    all_time_actions_dual.fill_(float_nan)
                    all_time_actions_left.fill_(float_nan)
                    all_time_actions_right.fill_(float_nan)
                    
                # Reset touch tracking
                acc_touched_left = set()
                acc_touched_right = set()
                    
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
    parser.add_argument('--command_sequence', action='store', type=str, default=None, help='Scheduler: Task sequence (e.g. ICLR)')
    parser.add_argument('--task_durations', action='store', type=str, default=None, help='Scheduler: JSON string of durations')
    parser.add_argument('--color_sequence', action='store', type=str, default=None, help='Color sequence (e.g. rrgbrrgbrr)')
    parser.add_argument('--episode_len', action='store', type=int, default=None, help='Override task-specific episode length')

    args = parser.parse_args()
    main(args)
