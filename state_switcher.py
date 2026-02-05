
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
import csv
import collections
import matplotlib.pyplot as plt
import numpy as np
import torch
from einops import rearrange
import IPython
import cv2
from PIL import Image
from torchvision import transforms

# --- TRAIN STATE CLASSIFIER IMPORTS ---
# Assuming train_state_classifier.py is in the same directory
from train_state_classifier import DualStateClassifier, STATE_HOLD, STATE_INDEP, STATE_COOP, STATE_NAMES

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
    """Return dict of sets of cube indices contacted by each gripper (based on link body names)."""
    touched = {'left': set(), 'right': set()}
    for i in range(physics.data.ncon):
        id1, id2 = physics.data.contact[i].geom1, physics.data.contact[i].geom2
        
        for g_id, o_id in [(id1, id2), (id2, id1)]:
            # Link body check (much more robust than geom names)
            b_id = physics.model.geom_bodyid[g_id]
            b_name = physics.model.id2name(b_id, 'body')
            if b_name is None: continue
            
            arm = None
            if b_name in ['l_link7', 'l_link8']: arm = 'left'
            elif b_name in ['r_link7', 'r_link8']: arm = 'right'
            
            if arm:
                o_name = physics.model.id2name(o_id, 'geom')
                if o_name and o_name.startswith('cube_'):
                    try:
                        c_idx = int(o_name.split('_')[1])
                        touched[arm].add(c_idx)
                    except: pass
    return touched

def get_grasped_cubes(physics):
    """Return dict of sets of cube indices currently grasped (contact with multiple fingers)."""
    grasped = {'left': set(), 'right': set()}
    # Track which fingers touch which cube
    finger_hits = {'left': collections.defaultdict(set), 'right': collections.defaultdict(set)}
    
    for i in range(physics.data.ncon):
        id1, id2 = physics.data.contact[i].geom1, physics.data.contact[i].geom2
        for g_id, o_id in [(id1, id2), (id2, id1)]:
            b_id = physics.model.geom_bodyid[g_id]
            b_name = physics.model.id2name(b_id, 'body')
            if b_name is None: continue
            
            arm, side = None, None
            if b_name == 'l_link7': arm, side = 'left', '1'
            elif b_name == 'l_link8': arm, side = 'left', '2'
            elif b_name == 'r_link7': arm, side = 'right', '1'
            elif b_name == 'r_link8': arm, side = 'right', '2'
            
            if arm:
                o_name = physics.model.id2name(o_id, 'geom')
                if o_name and o_name.startswith('cube_'):
                    try:
                        c_idx = int(o_name.split('_')[1])
                        finger_hits[arm][c_idx].add(side)
                        # print(f"DEBUG: Contact {arm} side {side} with cube_{c_idx}")
                    except: pass
                
    for arm in ['left', 'right']:
        for c_idx, sides in finger_hits[arm].items():
            if len(sides) >= 2:
                grasped[arm].add(c_idx)
                # print(f"DEBUG: Cube {c_idx} detected as GRASPED by {arm} arm")
    return grasped

def get_proximity_cubes(physics, threshold=0.06):
    """Return set of cube indices within threshold distance of any gripper link."""
    nearby = set()
    gripper_bodies = ['l_link7', 'l_link8', 'r_link7', 'r_link8']
    gripper_xpos = []
    for bn in gripper_bodies:
        try:
            bid = physics.model.name2id(bn, 'body')
            gripper_xpos.append(physics.data.xpos[bid])
        except: pass
    
    if not gripper_xpos: return nearby
    
    for i in range(10): # Check all 10 potential cubes
        try:
            c_name = f'cube_{i}'
            c_bid = physics.model.name2id(c_name, 'body')
            c_xpos = physics.data.xpos[c_bid]
            
            # Check distance to any gripper
            for g_pos in gripper_xpos:
                dist = np.linalg.norm(g_pos - c_xpos)
                if dist < threshold:
                    nearby.add(i)
                    break 
        except: pass
    return nearby

def get_cubes_touching_targets(physics, target_geoms):
    """Return set of cube indices recursively touching any of the target geoms."""
    # 1. Gather all contacts
    contacts = []
    for i in range(physics.data.ncon):
        id1, id2 = physics.data.contact[i].geom1, physics.data.contact[i].geom2
        name1 = physics.model.id2name(id1, 'geom')
        name2 = physics.model.id2name(id2, 'geom')
        if name1 and name2:
            contacts.append((name1, name2))

    # 2. Find base objects touching targets
    touching_targets = set()
    for n1, n2 in contacts:
        for a, b in [(n1, n2), (n2, n1)]:
            if a in target_geoms and b.startswith('cube_'):
                try:
                    c_idx = int(b.split('_')[1])
                    touching_targets.add(c_idx)
                except: pass

    # 3. Propagate (Transitive Closure) for stacked objects
    changed = True
    while changed:
        changed = False
        for n1, n2 in contacts:
            c1_idx = -1
            c2_idx = -1
            
            if n1.startswith('cube_'):
                try: c1_idx = int(n1.split('_')[1])
                except: pass
            
            if n2.startswith('cube_'):
                try: c2_idx = int(n2.split('_')[1])
                except: pass
                
            if c1_idx != -1 and c2_idx != -1:
                # If one is touching, the other is too
                if c1_idx in touching_targets and c2_idx not in touching_targets:
                    touching_targets.add(c2_idx)
                    changed = True
                elif c2_idx in touching_targets and c1_idx not in touching_targets:
                    touching_targets.add(c1_idx)
                    changed = True
                    
    return touching_targets

def get_cubes_in_goal(physics):
    return get_cubes_touching_targets(physics, {'goal_plate'})

def get_cubes_on_cushion(physics):
    return get_cubes_touching_targets(physics, {'cushion1'})


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

    # Auto-resolve checkpoint path
    real_ckpt_path = None
    if os.path.isfile(ckpt_dir):
        real_ckpt_path = ckpt_dir
    elif ckpt_dir.endswith('.ckpt'):
        # Try appending _seed_0 if not found
        candidate = ckpt_dir.replace('.ckpt', '_seed_0.ckpt')
        if os.path.isfile(candidate):
            real_ckpt_path = candidate
            print(f"Resolving {ckpt_dir} -> {real_ckpt_path}")
    
    if real_ckpt_path:
        # User provided a direct file path
        ckpt_path = real_ckpt_path
        parent_dir = os.path.dirname(real_ckpt_path)
        stats_path = os.path.join(parent_dir, 'dataset_stats.pkl')
    else:
        # User provided a directory, defaulting to policy_best.ckpt
        stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
        ckpt_path = os.path.join(ckpt_dir, 'policy_best.ckpt')

    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)
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


def load_state_classifier(ckpt_path, device='cuda'):
    print(f"Loading State Classifier from {ckpt_path}...")
    model = DualStateClassifier(num_classes=3)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    model.eval()
    return model

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

    # --- Mode Classifier ---
    print(f"Loading State Classifier from {args.state_ckpt}...")
    state_classifier = load_state_classifier(args.state_ckpt)
    
    # Classifier Transforms
    cls_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


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
    if sys.stdin.isatty():
        old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        print("Ready. Press 'q' to quit.")
    else:
        print("Not a TTY, manual control disabled")
    
    # Init Mode
    current_mode = MODE_COOP
    
    print("\n\nReady!")
    print("Press 'q' to quit")
    
    # State tracking for chunk execution
    step_in_chunk = 0
    current_action_chunk_dual = None
    current_action_chunk_left = None
    current_action_chunk_right = None
    
    # Initialize Temporal Aggregation Logic
    temporal_agg = True
    if args.no_temporal_agg:
        temporal_agg = False
    
    num_queries = args.chunk_size
    float_nan = float('nan')
    all_time_actions_dual = torch.full([max_timesteps, max_timesteps+num_queries, 14], float_nan).cuda()
    all_time_actions_left = torch.full([max_timesteps, max_timesteps+num_queries, 7], float_nan).cuda()
    all_time_actions_right = torch.full([max_timesteps, max_timesteps+num_queries, 7], float_nan).cuda()

    # Home Pose
    home_pose = np.zeros(14)
    home_pose[:6] = START_ARM_POSE[:6]
    home_pose[6] = START_ARM_POSE[6]
    home_pose[7:13] = START_ARM_POSE[8:14]
    home_pose[13] = START_ARM_POSE[14]

    try:
        t = 0
        video_frames = []
        episode_count = 0
        
        # Stats
        episode_returns = []
        highest_rewards = []
        current_episode_rewards = []
        
        # Removal Tracking
        acc_touched_left = set()
        acc_touched_right = set()
        pending_removal = set()
        
        # State Tracking Initialization
        prev_plan_l_state = None
        prev_plan_r_state = None
        state_history_l = collections.deque(maxlen=10)
        state_history_r = collections.deque(maxlen=10)
        
        while True:
            # Check input
            key = None
            if old_settings:
                key = get_key()
            if key == 'q':
                break
                
            # Render update matching imitate_episodes.py timing
            if onscreen_render:
                # Exit if window is closed
                if not plt.get_fignums():
                    print("Window closed. Exiting...")
                    break
                    
                image = env._physics.render(height=240, width=320, camera_id='top')
                plt_img.set_data(image)
                plt.pause(DT)

            if args.save_video:
                 # Render at 720p (1280x720) for video saving
                 video_frame = env._physics.render(height=720, width=1280, camera_id='top')
                 video_frames.append(video_frame) 
            
            # --- State Classifier Inference ---
            # 1. Get raw image for classifier
            raw_img_np = ts.observation['images']['top'] # (H,W,C)
            # 2. Transform
            raw_pil = Image.fromarray(raw_img_np.astype('uint8'))
            cls_input = cls_transform(raw_pil).unsqueeze(0).cuda()
            
            # 3. Predict
            with torch.no_grad():
                out_l, out_r = state_classifier(cls_input)
                _, pred_l = torch.max(out_l, 1)
                _, pred_r = torch.max(out_r, 1)
                
                s_l = pred_l.item()
                s_r = pred_r.item()
            
            # 4. Determine Execution Mode
            # --- State Smoothing (Hysteresis) ---
            state_history_l.append(s_l)
            state_history_r.append(s_r)
            
            # Majority Vote
            def get_majority(queue):
                if len(queue) == 0: return 0
                counts = collections.Counter(queue)
                return counts.most_common(1)[0][0]
            
            s_l_smooth = get_majority(state_history_l)
            s_r_smooth = get_majority(state_history_r)
            
            # Use smoothed states for planning
            plan_l_state = 'INDEP'
            plan_r_state = 'INDEP'
            
            if s_l_smooth == STATE_COOP and s_r_smooth == STATE_COOP:
                plan_l_state = 'COOP'
                plan_r_state = 'COOP'
            else:
                # Mixed or pure independent
                if s_l_smooth == STATE_COOP: plan_l_state = 'COOP'
                elif s_l_smooth == STATE_INDEP: plan_l_state = 'INDEP'
                else: plan_l_state = 'HOLD'
                
                if s_r_smooth == STATE_COOP: plan_r_state = 'COOP'
                elif s_r_smooth == STATE_INDEP: plan_r_state = 'INDEP'
                else: plan_r_state = 'HOLD'
            
            # Logging
            if t % 50 == 0:
                print(f"[Step {t}] Classifier: L={STATE_NAMES[s_l]} R={STATE_NAMES[s_r]} -> Plan: L={plan_l_state} R={plan_r_state}")
                if onscreen_render:
                    plt.title(f"Plan: L={plan_l_state} R={plan_r_state} (State: {STATE_NAMES[s_l]}/{STATE_NAMES[s_r]})")

            # --- Buffer Inheritance / State Tracking Logic (MOVED HERE) ---
            # Run BEFORE Inference/Aggregation to ensure buffers are ready for the switch.
            if args.inherit_temporal_buffer and temporal_agg:
                # LEFT transition checking
                if prev_plan_l_state is not None:
                     if prev_plan_l_state == 'INDEP' and plan_l_state == 'COOP':
                         input_actions = all_time_actions_left
                         mask_val = ~torch.isnan(input_actions)
                         input_safe = torch.nan_to_num(input_actions, nan=0.0)
                         denorm_l = input_safe * stats_left_torch['action_std'] + stats_left_torch['action_mean']
                         renorm_dual_l = (denorm_l - stats_dual_torch['action_mean'][:7]) / stats_dual_torch['action_std'][:7]
                         val_inherit = renorm_dual_l.clone()
                         val_inherit[~mask_val] = float('nan') 
                         all_time_actions_dual[:, :, :7] = val_inherit
                     
                     elif prev_plan_l_state == 'COOP' and plan_l_state == 'INDEP':
                         input_actions = all_time_actions_dual[:, :, :7]
                         mask_val = ~torch.isnan(input_actions)
                         input_safe = torch.nan_to_num(input_actions, nan=0.0)
                         denorm_dual_l = input_safe * stats_dual_torch['action_std'][:7] + stats_dual_torch['action_mean'][:7]
                         renorm_l = (denorm_dual_l - stats_left_torch['action_mean']) / stats_left_torch['action_std']
                         val_inherit = renorm_l.clone()
                         val_inherit[~mask_val] = float('nan')
                         all_time_actions_left.copy_(val_inherit)
                
                # RIGHT transition checking
                if prev_plan_r_state is not None:
                     if prev_plan_r_state == 'INDEP' and plan_r_state == 'COOP':
                         input_actions = all_time_actions_right
                         mask_val = ~torch.isnan(input_actions)
                         input_safe = torch.nan_to_num(input_actions, nan=0.0)
                         denorm_r = input_safe * stats_right_torch['action_std'] + stats_right_torch['action_mean']
                         renorm_dual_r = (denorm_r - stats_dual_torch['action_mean'][7:]) / stats_dual_torch['action_std'][7:]
                         val_inherit = renorm_dual_r.clone()
                         val_inherit[~mask_val] = float('nan')
                         all_time_actions_dual[:, :, 7:] = val_inherit

                     elif prev_plan_r_state == 'COOP' and plan_r_state == 'INDEP':
                         input_actions = all_time_actions_dual[:, :, 7:]
                         mask_val = ~torch.isnan(input_actions)
                         input_safe = torch.nan_to_num(input_actions, nan=0.0)
                         denorm_dual_r = input_safe * stats_dual_torch['action_std'][7:] + stats_dual_torch['action_mean'][7:]
                         renorm_r = (denorm_dual_r - stats_right_torch['action_mean']) / stats_right_torch['action_std']
                         val_inherit = renorm_r.clone()
                         val_inherit[~mask_val] = float('nan')
                         all_time_actions_right.copy_(val_inherit)

            
            obs = ts.observation
            qpos_numpy = np.array(obs['qpos'])
            
            with torch.inference_mode():
                # 1. Query Dual (if needed by ANY arm)
                if plan_l_state == 'COOP' or plan_r_state == 'COOP':
                     # Prepare input for Dual Policy
                    qpos_numpy_dual = qpos_numpy.copy()
                      
                     # --- GHOST ARM LOGIC (Corrected) ---
                     # Previously we forced the non-coop arm to HOME_POSE.
                     # This caused jumps because the Dual Policy saw "Arm Teleported to Home" and reacted.
                     # CORRECT LOGIC: Pass the REAL qpos of the non-coop arm. 
                     # The policy should be robust enough, or at least it won't see a teleport.
                     # So we DO NOT mask the input anymore.
                     # -------------------------------------------

                    qpos = pre_process_dual(qpos_numpy_dual)
                    qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                    curr_image = get_image_dual(ts, camera_names)
                      
                    action_chunk = policy_dual(qpos, curr_image) # [1, chunk_size, 14]
                    if temporal_agg:
                        all_time_actions_dual[[t], t:t+num_queries] = action_chunk
                    else:
                        current_action_chunk_dual = action_chunk.squeeze(0).cpu().numpy()


                # 2. Query Independent Left (if needed)
                if plan_l_state == 'INDEP':
                     qpos_left_numpy = qpos_numpy[:7]
                     qpos_left = pre_process_left(qpos_left_numpy)
                     qpos_left = torch.from_numpy(qpos_left).float().cuda().unsqueeze(0)
                     curr_image_left = get_image_independent(ts, camera_names, 'left')
                     
                     action_chunk_l = policy_left(qpos_left, curr_image_left)
                     if temporal_agg:
                         all_time_actions_left[[t], t:t+num_queries] = action_chunk_l
                     else:
                         current_action_chunk_left = action_chunk_l.squeeze(0).cpu().numpy()
                
                # 3. Query Independent Right (if needed)
                if plan_r_state == 'INDEP':
                     qpos_right_numpy = qpos_numpy[7:14]
                     qpos_right = pre_process_right(qpos_right_numpy)
                     qpos_right = torch.from_numpy(qpos_right).float().cuda().unsqueeze(0)
                     curr_image_right = get_image_independent(ts, camera_names, 'right')
                     
                     action_chunk_r = policy_right(qpos_right, curr_image_right)
                     if temporal_agg:
                         all_time_actions_right[[t], t:t+num_queries] = action_chunk_r
                     else:
                         current_action_chunk_right = action_chunk_r.squeeze(0).cpu().numpy()
                             
                
                # --- 1. Get Left Action ---
                if plan_l_state == 'COOP':
                    # Use Dual Output (Left Slice)
                    if temporal_agg:
                        actions_for_curr_step = all_time_actions_dual[:, t]
                        # FIX: Only check if Left Side (0-7) is valid.
                        # Inherited buffer might have Right Side as NaN.
                        actions_populated = torch.all(~torch.isnan(actions_for_curr_step[:, :7]), axis=1)
                        actions_for_curr_step = actions_for_curr_step[actions_populated]
                        k = 0.01
                        weights_len = len(actions_for_curr_step)
                        exp_weights = np.exp(-k * (weights_len - 1 - np.arange(weights_len)))
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                        raw_action_dual = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                        raw_action_dual = raw_action_dual.squeeze(0).cpu().numpy()
                        current_raw_action_l = raw_action_dual[:7]
                    else:
                        current_raw_action_l = current_action_chunk_dual[step_in_chunk][:7]
                    
                elif plan_l_state == 'INDEP':
                    # Independent Mode for Left
                    if temporal_agg:
                        actions_for_curr_step_l = all_time_actions_left[:, t]
                        actions_populated_l = torch.all(~torch.isnan(actions_for_curr_step_l), axis=1)
                        actions_for_curr_step_l = actions_for_curr_step_l[actions_populated_l]
                        
                        if len(actions_for_curr_step_l) == 0:
                             actions_for_curr_step_l = all_time_actions_left[:, t] 
                             
                        k = 0.01
                        weights_len_l = len(actions_for_curr_step_l)
                        exp_weights_l = np.exp(-k * (weights_len_l - 1 - np.arange(weights_len_l)))
                        exp_weights_l = exp_weights_l / exp_weights_l.sum()
                        exp_weights_l = torch.from_numpy(exp_weights_l).cuda().unsqueeze(dim=1)
                        raw_action_l = (actions_for_curr_step_l * exp_weights_l).sum(dim=0, keepdim=True)
                        current_raw_action_l = raw_action_l.squeeze(0).cpu().numpy()
                    else:
                        current_raw_action_l = current_action_chunk_left[step_in_chunk]
                else:
                    # HOLD mode - no raw action needed
                    current_raw_action_l = None

                # --- 2. Get Right Action ---
                if plan_r_state == 'COOP':
                    # Use Dual Output (Right Slice)
                    if temporal_agg:
                         # Re-calculate Dual (redundant if Left was also Coop, but safe)
                        actions_for_curr_step = all_time_actions_dual[:, t]
                        # FIX: Only check if Right Side (7-14) is valid.
                        actions_populated = torch.all(~torch.isnan(actions_for_curr_step[:, 7:]), axis=1)
                        actions_for_curr_step = actions_for_curr_step[actions_populated]
                        k = 0.01
                        weights_len = len(actions_for_curr_step)
                        exp_weights = np.exp(-k * (weights_len - 1 - np.arange(weights_len)))
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                        raw_action_dual = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                        raw_action_dual = raw_action_dual.squeeze(0).cpu().numpy()
                        current_raw_action_r = raw_action_dual[7:]
                    else:
                        current_raw_action_r = current_action_chunk_dual[step_in_chunk][7:]
                elif plan_r_state == 'INDEP':
                    # Independent Mode for Right
                    if temporal_agg:
                        actions_for_curr_step_r = all_time_actions_right[:, t]
                        actions_populated_r = torch.all(~torch.isnan(actions_for_curr_step_r), axis=1)
                        actions_for_curr_step_r = actions_for_curr_step_r[actions_populated_r]
                         
                        k = 0.01
                        weights_len_r = len(actions_for_curr_step_r)
                        
                        if weights_len_r == 0:
                             current_raw_action_r = current_action_chunk_right[step_in_chunk]
                        else:
                             exp_weights_r = np.exp(-k * (weights_len_r - 1 - np.arange(weights_len_r)))
                             exp_weights_r = exp_weights_r / exp_weights_r.sum()
                             exp_weights_r = torch.from_numpy(exp_weights_r).cuda().unsqueeze(dim=1)
                             raw_action_r = (actions_for_curr_step_r * exp_weights_r).sum(dim=0, keepdim=True)
                             current_raw_action_r = raw_action_r.squeeze(0).cpu().numpy()
                    else:
                        current_raw_action_r = current_action_chunk_right[step_in_chunk]
                else:
                    # HOLD mode - no raw action needed
                    current_raw_action_r = None

                # --- 3. Denormalize & Combine ---
                # Left
                if plan_l_state == 'COOP':
                     action_l = current_raw_action_l * stats_dual['action_std'][:7] + stats_dual['action_mean'][:7]
                elif plan_l_state == 'INDEP':
                     action_l = current_raw_action_l * stats_left['action_std'] + stats_left['action_mean']
                else:
                     # HOLD Mode: Maintain current joint position
                     action_l = qpos_numpy[:7]
                
                # Right
                if plan_r_state == 'COOP':
                     action_r = current_raw_action_r * stats_dual['action_std'][7:] + stats_dual['action_mean'][7:]
                elif plan_r_state == 'INDEP':
                     action_r = current_raw_action_r * stats_right['action_std'] + stats_right['action_mean']
                else:
                     # HOLD Mode: Maintain current joint position
                     action_r = qpos_numpy[7:14]
                
                # Combine
                action = np.concatenate([action_l, action_r])
                target_qpos = action
                
                # --- Safety Clamp REMOVED for debugging ---
                diff = target_qpos - qpos_numpy
                max_diff = np.max(np.abs(diff))
                if max_diff > 0.1:
                    print(f"\n[Step {t}] LARGE JUMP DETECTED! Max diff: {max_diff:.4f}")
                    # Find which joints are jumping
                    jump_indices = np.where(np.abs(diff) > 0.1)[0]
                    print(f"  Jumping Joints indices: {jump_indices}")
                    print(f"  Plan State: L={plan_l_state}, R={plan_r_state}")
                    
                    # Debug Buffer
                    if plan_l_state == 'INDEP':
                        print(f"  Left Buffer Count (approx): {len(actions_for_curr_step_l) if 'actions_for_curr_step_l' in locals() else 'N/A'}")
                        # Check normalization
                        # print(f"  Raw Action L (norm): {current_raw_action_l[:3]}")
                        # print(f"  Action L (denorm): {action_l[:3]}")
                        
                    if plan_l_state == 'COOP' or plan_r_state == 'COOP':
                         print(f"  Dual Buffer Count (approx): {len(actions_for_curr_step) if 'actions_for_curr_step' in locals() else 'N/A'}")

                
                # Update Previous States (Moved to top of loop logic)
                prev_plan_l_state = plan_l_state
                prev_plan_r_state = plan_r_state

            ts = env.step(target_qpos)
            current_episode_rewards.append(ts.reward)
            
            # --- Object Removal Logic (Continuous) ---
            # 1. Track Touches (if needed for debugging, but we mostly care about 'currently in hand')
            new_touches = get_touched_cubes_per_arm(env.physics)
            acc_touched_left.update(new_touches['left'])
            acc_touched_right.update(new_touches['right'])
            
            # 2. Check candidates for removal (Goal or Cushion)
            in_goal = get_cubes_in_goal(env.physics)
            on_cushion = get_cubes_on_cushion(env.physics)
            pending_removal.update(in_goal)
            pending_removal.update(on_cushion)
            
            # 3. Remove if NOT protected (touched/grasped/nearby)
            if pending_removal:
                 grasped = get_grasped_cubes(env.physics)
                 nearby = get_proximity_cubes(env.physics)
                 currently_touching = get_touched_cubes_per_arm(env.physics)
                 
                 protected_any = (grasped['left'] | grasped['right'] | 
                                  currently_touching['left'] | currently_touching['right'] | 
                                  nearby)
                 
                 to_remove = pending_removal - protected_any
                 if to_remove:
                     print(f"[Step {t}] Auto-Removing objects: {to_remove}")
                     remove_cubes(env.physics, list(to_remove))
                     pending_removal -= to_remove
            
            step_in_chunk += 1
            t += 1
            
            # Reset logic matching imitate_episodes num_rollouts loop
            if t >= max_timesteps:
                print("Episode finished. Resetting...")
                
                if args.save_video and len(video_frames) > 0:
                     video_path = f'sim_state_switch_ep{episode_count}.mp4'
                     target_w, target_h = 1280, 720
                     fps = 30
                     out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (target_w, target_h))
                     for frame in video_frames:
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

                # Reset Removal Tracking
                acc_touched_left.clear()
                acc_touched_right.clear()
                pending_removal.clear()
                
                # Reset State Tracking
                prev_plan_l_state = None
                prev_plan_r_state = None
                state_history_l.clear()
                state_history_r.clear()

                if episode_count >= args.num_rollouts:
                    break

                ts = reset_with_new_pose()
                t = 0
                step_in_chunk = 0
                
                # Reset temp buffers
                if temporal_agg:
                    all_time_actions_dual.fill_(float_nan)
                    all_time_actions_left.fill_(float_nan)
                    all_time_actions_right.fill_(float_nan)
                    
        # Summary
        avg_return = np.mean(episode_returns)
        print(f"\nEvaluation Finished.")
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
    parser.add_argument('--state_ckpt', action='store', type=str, required=True, help='Path to trained state classifier')
    
    parser.add_argument('--policy_class', action='store', type=str, default='ACT')
    parser.add_argument('--kl_weight', action='store', type=int, default=10)
    parser.add_argument('--chunk_size', action='store', type=int, default=100)
    parser.add_argument('--hidden_dim', action='store', type=int, default=512)
    parser.add_argument('--dim_feedforward', action='store', type=int, default=3200)
    parser.add_argument('--lr', action='store', type=float, default=1e-5)
    
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--save_video', action='store_true', help='Save execution video to mp4')
    parser.add_argument('--num_rollouts', action='store', type=int, default=1, help='Number of evaluation episodes')
    parser.add_argument('--color_sequence', action='store', type=str, default=None, help='Color sequence (e.g. rrgbrrgbrr)')
    parser.add_argument('--episode_len', action='store', type=int, default=None, help='Override task-specific episode length')
    parser.add_argument('--inherit_temporal_buffer', action='store_true', help='Inherit temporal aggregation buffer on switch to prevent jerk')
    parser.add_argument('--warmup_steps', action='store', type=int, default=0, help='Number of steps to run independent policy in background before switch')
    parser.add_argument('--no_temporal_agg', action='store_true', help='Disable temporal aggregation')
    
    args = parser.parse_args()
    main(args)
