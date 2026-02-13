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
import csv
import collections
from piper_constants import DT

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
                
                # Unified C duration (No split requested)
                if len_assembly == 0 and len_place == 0 and 'C' in self.config:
                    total_c = self.config['C']
                    len_assembly = total_c
                    len_place = 0
                
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
                
                # Phase 2: Placement (Base stays COOP, Free becomes INDEP)
                end_place = end_assembly + len_place
                # Global Mode remains COOP until end_place
                self.mode_schedule[end_assembly] = self.MODE_COOP
                self.mode_schedule[end_place] = self.MODE_INDEPENDENT
                
                
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

                if base_arm == 'left':
                    # Left blocked (Base) -> Continues COOP
                    # Right free (Top) -> Goes INDEP immediately
                    self.timeline_left.append({'start': end_assembly, 'end': end_place, 'type': 'COOP', 'info': 'Phase 2 (Place-Base)'})
                    time_l = end_place
                    time_r = end_assembly # Right free immediately
                else:
                    # Right blocked (Base) -> Continues COOP
                    # Left free (Top) -> Goes INDEP immediately
                    self.timeline_right.append({'start': end_assembly, 'end': end_place, 'type': 'COOP', 'info': 'Phase 2 (Place-Base)'})
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
    
    orig_shape = images.shape
    if len(orig_shape) == 5:
        b, n_cam, c, h, w = orig_shape
        images = images.view(b * n_cam, c, h, w)
        
    B, C, H, W = images.shape
    
    if arm == 'left':
        strip_start = W - strip_width
        strip_end = W
    else:
        strip_start = 0
        strip_end = strip_width
        
    strip = images[:, :, :, strip_start:strip_end]
    
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

def get_image_dual(ts, camera_names):
    curr_images = []
    for cam_name in camera_names:
        curr_image_np = ts.observation['images'][cam_name].copy()
        curr_image_t = torch.from_numpy(curr_image_np).permute(2, 0, 1).unsqueeze(0).float().cuda() / 255.0
        curr_images.append(curr_image_t.squeeze(0))
    curr_image = torch.stack(curr_images, dim=0).unsqueeze(0)
    return curr_image

def get_image_independent(ts, camera_names, arm):
    curr_images = []
    for cam_name in camera_names:
        curr_image_np = ts.observation['images'][cam_name].copy()
        h, w, c = curr_image_np.shape
        curr_image_t = torch.from_numpy(curr_image_np).permute(2, 0, 1).unsqueeze(0).float().cuda() / 255.0
        
        if arm == 'left':
             curr_image_t = curr_image_t[:, :, :, :w//2]
             curr_image_t = apply_torch_rgb_mask(curr_image_t, strip_width=40, arm='left')
        else:
             curr_image_t = curr_image_t[:, :, :, w//2:]
             curr_image_t = apply_torch_rgb_mask(curr_image_t, strip_width=40, arm='right')
             
        curr_images.append(curr_image_t.squeeze(0))
        
    curr_image = torch.stack(curr_images, dim=0).unsqueeze(0)
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
                    except: pass
                
    for arm in ['left', 'right']:
        for c_idx, sides in finger_hits[arm].items():
            if len(sides) >= 2:
                grasped[arm].add(c_idx)
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
            
            # Check dist to any gripper part
            for gx in gripper_xpos:
                dist = np.linalg.norm(c_xpos - gx)
                if dist < threshold:
                    nearby.add(i)
                    break
        except: pass
    return nearby

def get_cubes_on_target_geoms(physics, target_names):
    """Return set of cube indices contacting any of the specified geoms."""
    on_target = set()
    if isinstance(target_names, str):
        target_names = [target_names]
        
    for i in range(physics.data.ncon):
        id1, id2 = physics.data.contact[i].geom1, physics.data.contact[i].geom2
        name1 = physics.model.id2name(id1, 'geom')
        name2 = physics.model.id2name(id2, 'geom')
        
        if name1 is None or name2 is None: continue
        
        for n1, n2 in [(name1, name2), (name2, name1)]:
            # Check if n1 is in our target list (exact match or startswith for robustness?)
            # User said "cushion1_mesh", sim uses "cushion1" probably. 
            # Let's match exact or substring if "mesh" is involved?
            # Safe bet: Check if n1 is in list.
            if n1 in target_names:
                if n2.startswith('cube_'):
                    try:
                        c_idx = int(n2.split('_')[1])
                        on_target.add(c_idx)
                    except: pass
    return on_target

def is_at_home(current_qpos, home_qpos, threshold=0.45):
    """Check if arm is close to home pose."""
    diff = np.abs(current_qpos - home_qpos)
    return np.max(diff) < threshold

def quaternion_multiply(q1, q2):
    """Multiply two quaternions. q1 * q2"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*z2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def quaternion_inverse(q):
    """Inverse of quaternion [w, x, y, z]"""
    return np.array([q[0], -q[1], -q[2], -q[3]]) / np.dot(q, q)

def rotate_vector_by_quaternion(v, q):
    """Rotate vector v by quaternion q."""
    q_vec = np.array([0, v[0], v[1], v[2]])
    q_inv = quaternion_inverse(q)
    temp = quaternion_multiply(q, q_vec)
    result = quaternion_multiply(temp, q_inv)
    return result[1:]

def apply_magnet_logic(physics, magnetized_pairs, color_sequence=None):
    """
    Visual-Only Stacking Logic (Magnet).
    """
    # 1. Identify Green and Blue cubes
    if color_sequence:
        greens = [i for i, c in enumerate(color_sequence) if c == 'g']
        blues = [i for i, c in enumerate(color_sequence) if c == 'b']
    else:
        greens = [8]
        blues = [9]

    # 2. Check for new magnetizations
    threshold = 0.08 # 8cm
    
    for g_idx in greens:
        if g_idx in magnetized_pairs: continue
        
        try:
            g_body_id = physics.model.name2id(f'cube_{g_idx}', 'body')
            g_pos = physics.data.xpos[g_body_id].copy()
            g_quat = physics.data.xquat[g_body_id].copy()
            
            for b_idx in blues:
                 b_body_id = physics.model.name2id(f'cube_{b_idx}', 'body')
                 b_pos = physics.data.xpos[b_body_id].copy()
                 b_quat = physics.data.xquat[b_body_id].copy()
                 
                 dist = np.linalg.norm(g_pos - b_pos)
                 
                 if dist < threshold:
                     print(f"Magnet Triggered: Green {g_idx} -> Blue {b_idx} (Dist: {dist:.4f})")
                     
                     global_offset = g_pos - b_pos
                     b_quat_inv = quaternion_inverse(b_quat)
                     rel_pos = rotate_vector_by_quaternion(global_offset, b_quat_inv)
                     rel_quat = quaternion_multiply(b_quat_inv, g_quat)
                     
                     magnetized_pairs[g_idx] = {
                         'blue_idx': b_idx,
                         'rel_pos': rel_pos,
                         'rel_quat': rel_quat
                     }
                     
                     # Disable Collision
                     g_geom_id = physics.model.name2id(f'cube_{g_idx}', 'geom')
                     physics.model.geom_contype[g_geom_id] = 0
                     physics.model.geom_conaffinity[g_geom_id] = 0
                     
                     break
        except Exception as e:
            print(f"Magnet check error: {e}")
            pass

    # 3. Apply updates
    for g_idx, data in magnetized_pairs.items():
        try:
            b_idx = data['blue_idx']
            rel_pos = data['rel_pos']
            rel_quat = data['rel_quat']
            
            b_body_id = physics.model.name2id(f'cube_{b_idx}', 'body')
            b_pos = physics.data.xpos[b_body_id].copy()
            b_quat = physics.data.xquat[b_body_id].copy()
            
            global_offset = rotate_vector_by_quaternion(rel_pos, b_quat)
            target_pos = b_pos + global_offset
            target_quat = quaternion_multiply(b_quat, rel_quat)
            
            start_idx = physics.model.name2id(f'cube_{g_idx}_joint', 'joint')
            qpos_adr = physics.model.jnt_qposadr[start_idx]
            
            physics.data.qpos[qpos_adr : qpos_adr+3] = target_pos
            physics.data.qpos[qpos_adr+3 : qpos_adr+7] = target_quat
            
            # Zero out velocity
            qvel_adr = physics.model.jnt_dofadr[start_idx]
            physics.data.qvel[qvel_adr : qvel_adr+6] = 0.0

        except Exception as e:
            print(f"Magnet apply error for G{g_idx}: {e}")
            pass

def reset_magnet_logic(physics, color_sequence=None):
    """Restore collision for ALL cubes at reset."""
    for i in range(10):
        try:
             geom_id = physics.model.name2id(f'cube_{i}', 'geom')
             physics.model.geom_contype[geom_id] = 1
             physics.model.geom_conaffinity[geom_id] = 1
        except: pass

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

    if os.path.isfile(ckpt_dir):
        # User provided a direct file path (e.g. policy_epoch_1000.ckpt)
        ckpt_path = ckpt_dir
        parent_dir = os.path.dirname(ckpt_dir)
        stats_path = os.path.join(parent_dir, f'dataset_stats.pkl')
    else:
        # User provided a directory, defaulting to policy_best.ckpt
        stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
        ckpt_path = os.path.join(ckpt_dir, 'policy_best.ckpt')

    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

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

    # Import Globals from Sim Env (Moved up to fix UnboundLocalError)
    from piper_sim_env import MANYCUBES_COLORS, MANYCUBES_TASK_COUNT
    import piper_constants
    
    # Default Fallback (if not defined elsewhere)
    COLOR_SEQUENCE = list('rrgbrrgbrr') # Default if not provided

    # --- Sequence Loading Logic ---
    available_sequences = []
    if args.sequence_file:
        import csv
        if not os.path.isfile(args.sequence_file):
            print(f"Error: Sequence file {args.sequence_file} not found.")
            return
        
        with open(args.sequence_file, 'r', newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                # Filter for 'success' sequences (3rd column)
                if len(row) >= 3 and row[2] == 'success':
                    available_sequences.append({
                        'color_seq': list(row[0].lower()),
                        'commands': row[1]
                    })
        
        if not available_sequences:
            print(f"Error: No successful sequences found in {args.sequence_file}")
            return
        print(f"Loaded {len(available_sequences)} successful sequences from {args.sequence_file}")
    
    # Validation check for non-CSV mode
    if not args.sequence_file and not args.commands:
        print("Error: --commands is required unless --sequence_file is specified.")
        return

    from piper_constants import SIM_TASK_CONFIGS
    task_config = SIM_TASK_CONFIGS[task_name]
    camera_names = task_config['camera_names']

    # Initial command queue for scheduler (will be overridden in loop if using sequence_file)
    command_queue = list(args.commands) if args.commands else []
    
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
        stats_left = {
            'action_mean': stats_e2e['action_mean'][:7],
            'action_std': stats_e2e['action_std'][:7],
            'qpos_mean': stats_e2e['qpos_mean'][:7],
            'qpos_std': stats_e2e['qpos_std'][:7]
        }
        policy_right = None
        stats_right = {
            'action_mean': stats_e2e['action_mean'][7:],
            'action_std': stats_e2e['action_std'][7:],
            'qpos_mean': stats_e2e['qpos_mean'][7:],
            'qpos_std': stats_e2e['qpos_std'][7:]
        }
        
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
    # Initialize TaskScheduler
    # Assuming config for duration
    duration_config = {
        'I': args.step_i,
        'C_assembly': int(args.step_c * 0.6) if args.step_c else 300, 
        'C_place': int(args.step_c * 0.4) if args.step_c else 220,    
        'Single': 200      
    }

    # Override for E2E mode to prevent premature stop
    if args.ckpt_e2e:
        duration_config['I'] = 1000
        duration_config['C_assembly'] = 1000
        duration_config['C_place'] = 1000
        duration_config['Single'] = 1000
    
    scheduler = TaskScheduler(command_queue, duration_config)
    
    # Override max steps if user requested
    if args.max_timesteps:
        scheduler.max_timesteps = args.max_timesteps
        print(f"Overriding scheduler max timesteps to {args.max_timesteps}")

    scheduler.print_schedule()
    
    # Calculate required time limit from Scheduler
    time_limit = (scheduler.max_timesteps + 200) * DT 
    
    # Pass camera_names to avoid rendering default 5 cameras (huge speedup)
    import piper_constants
    if args.x_shift:
        piper_constants.MANYCUBES_CONFIG['x_shift'] = args.x_shift
        piper_constants.MANYCUBES_CONFIG['x_shift_start_idx'] = args.x_shift_start_idx
        print(f"Applying x-shift: {args.x_shift} starting from index {args.x_shift_start_idx}")
    
    # Environment Setup
    from piper_sim_env import make_sim_env
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
    
    # Calculate a safe MAX_BUFFER_STEPS
    # Since sequences can vary if loaded from CSV, we use a large enough default or user override
    max_steps_allowed = args.max_timesteps if args.max_timesteps else 3500
    MAX_BUFFER_STEPS = max_steps_allowed + 500
    
    float_nan = float('nan')
    all_time_actions_dual = torch.full([MAX_BUFFER_STEPS, MAX_BUFFER_STEPS+num_queries, 14], float_nan).cuda()
    all_time_actions_left = torch.full([MAX_BUFFER_STEPS, MAX_BUFFER_STEPS+num_queries, 7], float_nan).cuda()
    all_time_actions_right = torch.full([MAX_BUFFER_STEPS, MAX_BUFFER_STEPS+num_queries, 7], float_nan).cuda()

    episode_count = 0
    success_count = 0
    total_rewards = []
    
    # Prepare Home Pose for Holding
    try:
        from piper_constants import START_ARM_POSE
        home_pose = np.array(START_ARM_POSE)
    except ImportError:
        print("Warning: START_ARM_POSE not found. Using default zeros for Hold.")
        home_pose = np.zeros(14)
    
    # 2-second delay logic for goal plate
    # DT is imported from piper_constants at top of file
    GOAL_DELAY_STEPS = int(2.0 / DT) 

    try:
        while episode_count < args.num_rollouts:
            # --- Sequence Selection ---
            if available_sequences:
                # Randomly pick from successful sequences
                selected = np.random.choice(available_sequences)
                color_seq = selected['color_seq']
                command_queue = list(selected['commands'])
                # Override if CLI provided (user request: "flexible")
                if args.color_sequence:
                    color_seq = list(args.color_sequence.lower())
                if args.commands:
                    command_queue = list(args.commands)
                    
                print(f"Rollout {episode_count} | Loaded Sequence: {''.join(color_seq)}, Commands: {''.join(command_queue)}")
            else:
                # Use CLI values
                command_queue = list(args.commands)
                color_seq = list(args.color_sequence.lower()) if args.color_sequence else COLOR_SEQUENCE
            
            # Update Globals for simulation environment
            MANYCUBES_COLORS[0] = color_seq
            MANYCUBES_TASK_COUNT[0] = len(command_queue)
            
            # Re-initialize Scheduler for this sequence
            duration_config = {
                'I': args.step_i,
                'C_assembly': int(args.step_c * 0.6) if args.step_c else 300, 
                'C_place': int(args.step_c * 0.4) if args.step_c else 220,    
                'Single': 200      
            }
            if args.ckpt_e2e:
                duration_config['I'] = 1000
                duration_config['C_assembly'] = 1000
                duration_config['C_place'] = 1000
                duration_config['Single'] = 1000
                
            scheduler = TaskScheduler(command_queue, duration_config)
            if args.max_timesteps:
                scheduler.max_timesteps = args.max_timesteps
            
            # Reset tracking per episode
            touching_goal_start_step = {} 
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
            
            video_frames = []
            
            current_touched_cubes = set() # (Legacy global tracking for reward?)
            acc_touched_left = set()
            acc_touched_right = set()
            
            # State History for Smart HOLD
            last_active_l_state = None
            last_active_r_state = None
            
            # Magnet Tracking
            reset_magnet_logic(env.physics, color_seq)
            magnetized_pairs = {}
            
            current_mode = scheduler.get_mode_at_step(0)
            if args.ckpt_e2e:
                current_mode = MODE_COOP
            
            print(f"\nEpisode {episode_count} Started.")
            while True:
                def handle_mode_switch(t, from_mode, to_mode):
                    """Unified mode switch with inheritance for evaluation."""
                    nonlocal current_mode, step_in_chunk
                    if from_mode == to_mode: return
                    
                    print(f"[Step {t}] Switching Mode: {from_mode} -> {to_mode}")
                    current_mode = to_mode
                    step_in_chunk = 0 # Replan
                    
                    if not (args.inherit_temporal_buffer and temporal_agg):
                        return
                        
                    if to_mode == MODE_INDEPENDENT:
                        # Dual -> Indep (Left/Right)
                        input_actions = all_time_actions_dual
                        dual_mask_val = ~torch.isnan(input_actions)
                        input_actions_safe = torch.nan_to_num(input_actions, nan=0.0)
                        denorm_ac = input_actions_safe * stats_dual_torch['action_std'] + stats_dual_torch['action_mean']
                        
                        # Left
                        denorm_l = denorm_ac[:, :, :7]
                        renorm_l = (denorm_l - stats_left_torch['action_mean']) / stats_left_torch['action_std']
                        all_time_actions_left.copy_(renorm_l)
                        all_time_actions_left[~dual_mask_val[:, :, :7]] = float_nan
                        
                        # Right
                        denorm_r = denorm_ac[:, :, 7:]
                        renorm_r = (denorm_r - stats_right_torch['action_mean']) / stats_right_torch['action_std']
                        all_time_actions_right.copy_(renorm_r)
                        all_time_actions_right[~dual_mask_val[:, :, 7:]] = float_nan
                        
                    elif to_mode == MODE_COOP:
                        # Indep -> Dual
                        input_l = all_time_actions_left
                        input_r = all_time_actions_right
                        l_mask = ~torch.isnan(input_l)
                        r_mask = ~torch.isnan(input_r)
                        comb_mask = torch.cat([l_mask, r_mask], dim=2)
                        
                        denom_l = torch.nan_to_num(input_l, nan=0.0) * stats_left_torch['action_std'] + stats_left_torch['action_mean']
                        denom_r = torch.nan_to_num(input_r, nan=0.0) * stats_right_torch['action_std'] + stats_right_torch['action_mean']
                        denom_dual = torch.cat([denom_l, denom_r], dim=2)
                        
                        ren_dual = (denom_dual - stats_dual_torch['action_mean']) / stats_dual_torch['action_std']
                        all_time_actions_dual.copy_(ren_dual)
                        all_time_actions_dual.copy_(ren_dual)
                        all_time_actions_dual[~comb_mask] = float_nan


                # -------------------------------
                # Auto-Switching Logic via Scheduler
                # -------------------------------
                new_mode = scheduler.get_mode_at_step(t)
                if args.ckpt_e2e:
                     new_mode = MODE_COOP
                if new_mode != current_mode:
                    handle_mode_switch(t, current_mode, new_mode)
                
                # --- Per-Arm Inheritance for Overlap (Coop -> Indep while Global is still Coop) ---
                if args.inherit_temporal_buffer and scheduler and current_mode == MODE_COOP and new_mode == MODE_COOP:
                     # Check individual arm transitions
                     prev_l_state, _ = scheduler.get_arm_state(t-1, 'left')
                     curr_l_state, _ = scheduler.get_arm_state(t, 'left')
                     
                     if prev_l_state == 'COOP' and curr_l_state == 'INDEP':
                         print(f"[Step {t}] Inheriting temporal buffer Left (Dual -> Indep) [Overlap]")
                         # Inherit Left from Dual
                         input_dual = all_time_actions_dual
                         input_dual_l = input_dual[:, :, :7]
                         
                         mask_dual = ~torch.isnan(input_dual_l)
                         input_dual_safe = torch.nan_to_num(input_dual_l, nan=0.0)
                         
                         denorm_dual = input_dual_safe * stats_dual_torch['action_std'][:7] + stats_dual_torch['action_mean'][:7]
                         renorm_l = (denorm_dual - stats_left_torch['action_mean']) / stats_left_torch['action_std']
                         
                         val_inherit_l = renorm_l.clone()
                         val_inherit_l[~mask_dual] = float_nan
                         all_time_actions_left.copy_(val_inherit_l)

                     prev_r_state, _ = scheduler.get_arm_state(t-1, 'right')
                     curr_r_state, _ = scheduler.get_arm_state(t, 'right')
                     
                     if prev_r_state == 'COOP' and curr_r_state == 'INDEP':
                         print(f"[Step {t}] Inheriting temporal buffer Right (Dual -> Indep) [Overlap]")
                         # Inherit Right from Dual
                         input_dual = all_time_actions_dual
                         input_dual_r = input_dual[:, :, 7:]
                         
                         mask_dual = ~torch.isnan(input_dual_r)
                         input_dual_safe = torch.nan_to_num(input_dual_r, nan=0.0)
                         
                         denorm_dual = input_dual_safe * stats_dual_torch['action_std'][7:] + stats_dual_torch['action_mean'][7:]
                         renorm_r = (denorm_dual - stats_right_torch['action_mean']) / stats_right_torch['action_std']
                         
                         val_inherit_r = renorm_r.clone()
                         val_inherit_r[~mask_dual] = float_nan
                         all_time_actions_right.copy_(val_inherit_r)
                
                # Check End
                if t >= scheduler.max_timesteps:
                     print(f"[Step {t}] Reached End of Schedule.")
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
                     # Render at 720p (1280x720) for video saving
                     onscreen_cam = 'top'
                     video_frame_highres = env._physics.render(height=720, width=1280, camera_id=onscreen_cam)
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

                        if scheduler:
                             plan_l_state, _ = scheduler.get_arm_state(t, 'left')
                             plan_r_state, _ = scheduler.get_arm_state(t, 'right')
                             # if t < 5: print(f"DEBUG Step {t}: Scheduler says {plan_l_state}")
                        else:
                             plan_l_state = 'COOP' if current_mode == MODE_COOP else 'INDEP'
                             plan_r_state = 'COOP' if current_mode == MODE_COOP else 'INDEP'
                             
                        # Force COOP planning path if E2E (Single Policy)
                        # Force COOP planning path if E2E (Single Policy)
                        if args.ckpt_e2e:
                            plan_l_state = 'COOP'
                            plan_r_state = 'COOP'
                            # if t < 5: print(f"DEBUG Step {t}: Overrode to COOP")
                            
                        # Update Last Active State (before overriding or after? Before is better as 'intent')
                        if plan_l_state in ['COOP', 'INDEP']: last_active_l_state = plan_l_state
                        if plan_r_state in ['COOP', 'INDEP']: last_active_r_state = plan_r_state
                        
                        # --- Smart HOLD Logic (Simulation) ---
                        # In Evaluate, we don't have explicit HOLD from scheduler usually.
                        # But if we did, or if we want to force return home at end?
                        # For now, we only implement if plan says HOLD.
                        
                        if plan_l_state == 'HOLD':
                             if not is_at_home(qpos_numpy[:7], home_pose[:7]):
                                  if last_active_l_state: plan_l_state = last_active_l_state
                        
                        if plan_r_state == 'HOLD':
                             if not is_at_home(qpos_numpy[7:14], home_pose[7:14]):
                                  if last_active_r_state: plan_r_state = last_active_r_state

                        # Check for switching to trigger chunk align
                        mode_switch_detected = False
                        if 'prev_plan_l_state' in locals() and prev_plan_l_state != plan_l_state:
                             mode_switch_detected = True
                        if 'prev_plan_r_state' in locals() and prev_plan_r_state != plan_r_state:
                             mode_switch_detected = True
                        
                        prev_plan_l_state = plan_l_state
                        prev_plan_r_state = plan_r_state
                        
                        if mode_switch_detected and not temporal_agg:
                             step_in_chunk = 0

                        # 1. Query Dual (Coop)

                        # 1. Query Dual (Coop)
                        # NOTE: In eval, we assume 'policy_dual' handles COOP tasks.
                        if plan_l_state == 'COOP' or plan_r_state == 'COOP':
                            qpos = pre_process_dual(qpos_numpy)
                            qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                            curr_image = get_image_dual(ts, camera_names)
                            
                            action_chunk = policy_dual(qpos, curr_image)
                            # if t < 5: print(f"DEBUG Step {t}: Policy executed. Chunk shape: {action_chunk.shape}")
                            if temporal_agg:
                                all_time_actions_dual[[t], t:t+num_queries] = action_chunk
                            else:
                                current_action_chunk_dual = action_chunk.squeeze(0).cpu().numpy()

                        # 2. Query Independent Left
                        if plan_l_state == 'INDEP':
                             if policy_left: # Using separate policy
                                 qpos_left_numpy = qpos_numpy[:7]
                                 qpos_left = pre_process_left(qpos_left_numpy)
                                 qpos_left = torch.from_numpy(qpos_left).float().cuda().unsqueeze(0)
                                 curr_image_left = get_image_independent(ts, camera_names, 'left')
                                 action_chunk_l = policy_left(qpos_left, curr_image_left)
                                 
                                 if temporal_agg:
                                     all_time_actions_left[[t], t:t+num_queries] = action_chunk_l
                                 else:
                                     current_action_chunk_left = action_chunk_l.squeeze(0).cpu().numpy()
                             elif policy_independent_dual: # Using Unified Independent Policy
                                  # TODO: Support unified independent in Mixed Mode? 
                                  # For now assuming user provided --ckpt_left/--ckpt_right as per logic
                                  pass

                        # 3. Query Independent Right
                        if plan_r_state == 'INDEP':
                             if policy_right:
                                 qpos_right_numpy = qpos_numpy[7:14]
                                 qpos_right = pre_process_right(qpos_right_numpy)
                                 qpos_right = torch.from_numpy(qpos_right).float().cuda().unsqueeze(0)
                                 curr_image_right = get_image_independent(ts, camera_names, 'right')
                                 
                                 action_chunk_r = policy_right(qpos_right, curr_image_right)
                                 if temporal_agg:
                                     all_time_actions_right[[t], t:t+num_queries] = action_chunk_r
                                 else:
                                     current_action_chunk_right = action_chunk_r.squeeze(0).cpu().numpy()

                # --- Execute / Combine ---
                # Determine state again for execution (same as plan)
                # Determine state again for execution (same as plan)
                if scheduler:
                     l_state, _ = scheduler.get_arm_state(t, 'left')
                     r_state, _ = scheduler.get_arm_state(t, 'right')
                else:
                     l_state = 'COOP' if current_mode == MODE_COOP else 'INDEP'
                     r_state = 'COOP' if current_mode == MODE_COOP else 'INDEP'
                
                # Force COOP execution path if E2E (Single Policy)
                if args.ckpt_e2e:
                    l_state = 'COOP'
                    r_state = 'COOP'
                
                # ... get raw actions ...
                
                # --- Left ---
                if l_state == 'COOP':
                    if temporal_agg:
                        actions_for_curr_step = all_time_actions_dual[:, t]
                        actions_populated = torch.all(~torch.isnan(actions_for_curr_step), axis=1)
                        actions_for_curr_step = actions_for_curr_step[actions_populated]
                        # if t < 5: 
                        #      print(f"DEBUG Step {t}: Valid actions count: {len(actions_for_curr_step)}")
                        #      if len(actions_for_curr_step) > 0:
                        #           print(f"DEBUG Step {t}: Sample Action[0]: {actions_for_curr_step[0, :3].cpu().numpy()}...")
                        k = 0.01
                        weights_len = len(actions_for_curr_step)
                        exp_weights = np.exp(-k * (weights_len - 1 - np.arange(weights_len)))
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                        raw_action_dual = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                        raw_action_dual = raw_action_dual.squeeze(0).cpu().numpy()
                        current_raw_action_l = raw_action_dual[:7]
                    else:
                        safe_step = step_in_chunk 
                        if safe_step >= chunk_size: safe_step = 0
                        current_raw_action_l = current_action_chunk_dual[safe_step][:7]
                else:
                    if temporal_agg:
                        actions_for_curr_step_l = all_time_actions_left[:, t] # No offset in eval usually?
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
                        safe_step = step_in_chunk 
                        if safe_step >= chunk_size: safe_step = 0
                        current_raw_action_l = current_action_chunk_left[safe_step]

                # --- Right ---
                if r_state == 'COOP':
                    if temporal_agg:
                        # Re-calc dual (redundant but safe)
                        actions_for_curr_step = all_time_actions_dual[:, t]
                        actions_populated = torch.all(~torch.isnan(actions_for_curr_step), axis=1)
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
                        safe_step = step_in_chunk 
                        if safe_step >= chunk_size: safe_step = 0
                        current_raw_action_r = current_action_chunk_dual[safe_step][7:]
                else:
                    if temporal_agg:
                        actions_for_curr_step_r = all_time_actions_right[:, t]
                        actions_populated_r = torch.all(~torch.isnan(actions_for_curr_step_r), axis=1)
                        actions_for_curr_step_r = actions_for_curr_step_r[actions_populated_r]
                        if len(actions_for_curr_step_r) == 0:
                             actions_for_curr_step_r = all_time_actions_right[:, t]
                        k = 0.01
                        weights_len_r = len(actions_for_curr_step_r)
                        exp_weights_r = np.exp(-k * (weights_len_r - 1 - np.arange(weights_len_r)))
                        exp_weights_r = exp_weights_r / exp_weights_r.sum()
                        exp_weights_r = torch.from_numpy(exp_weights_r).cuda().unsqueeze(dim=1)
                        raw_action_r = (actions_for_curr_step_r * exp_weights_r).sum(dim=0, keepdim=True)
                        current_raw_action_r = raw_action_r.squeeze(0).cpu().numpy()
                    else:
                        safe_step = step_in_chunk 
                        if safe_step >= chunk_size: safe_step = 0
                        current_raw_action_r = current_action_chunk_right[safe_step]

                # --- Denorm ---
                if l_state == 'COOP':
                     action_l = current_raw_action_l * stats_dual['action_std'][:7] + stats_dual['action_mean'][:7]
                else:
                     action_l = current_raw_action_l * stats_left['action_std'] + stats_left['action_mean']
                
                if r_state == 'COOP':
                     action_r = current_raw_action_r * stats_dual['action_std'][7:] + stats_dual['action_mean'][7:]
                else:
                     action_r = current_raw_action_r * stats_right['action_std'] + stats_right['action_mean']
                
                action = np.concatenate([action_l, action_r])
                target_qpos = action
                # Apply Scheduler Holds
                hold_l = scheduler.should_hold(t, 'left')
                hold_r = scheduler.should_hold(t, 'right')
                
                if args.ckpt_e2e:
                    hold_l = False
                    hold_r = False
                
                if hold_l:
                    target_qpos[:7] = qpos_numpy[:7]
                if hold_r:
                    target_qpos[7:] = qpos_numpy[7:]

                ts = env.step(target_qpos)
                current_reward = env._task.get_reward(env.physics)
                
                # --- Magnet Logic ---
                apply_magnet_logic(env.physics, magnetized_pairs, color_seq)
                
                # Track touched cubes
                # Track touched cubes
                new_touches = get_touched_cubes_per_arm(env.physics)
                acc_touched_left.update(new_touches['left'])
                acc_touched_right.update(new_touches['right'])
                
                # --- Goal Plate & Cushion Removal Logic ---
                # Targets: Goal Plate + Cushions (User req: cushion1_mesh, cushion2_mesh)
                # --- Goal Plate & Cushion Removal Logic ---
                # Targets: Goal Plate + Cushions (User req: cushion1_mesh, cushion2_mesh)
                removal_targets = ['goal_plate', 'cushion1', 'cushion1_mesh', 'cushion2', 'cushion2_mesh']
                on_plate = get_cubes_on_target_geoms(env.physics, removal_targets)
                
                # Register NEW contacts (Latch logic: Once touched, timer starts and never resets)
                for c_idx in on_plate:
                    if c_idx not in touching_goal_start_step:
                         touching_goal_start_step[c_idx] = t
                
                # Check ALL pending removals (even if contact lost)
                # Delay: 1.0s = 50 steps
                REMOVAL_DELAY_STEPS = int(1.0 / DT)
                to_remove_goal = []
                
                # Check all tracked items
                for c_idx, start_t in list(touching_goal_start_step.items()):
                     elapsed = t - start_t
                     if elapsed > REMOVAL_DELAY_STEPS:
                         # Safety check: Don't remove if currently gripped! (e.g. still placing)
                         grasped = get_grasped_cubes(env.physics)
                         if c_idx not in grasped['left'] and c_idx not in grasped['right']:
                             to_remove_goal.append(c_idx)
                
                if to_remove_goal:
                    print(f"[Step {t}] Removing objects on Goal Plate > 2s: {to_remove_goal}")
                    remove_cubes(env.physics, to_remove_goal)
                    for c in to_remove_goal:
                         if c in touching_goal_start_step: del touching_goal_start_step[c]
                         # Also remove from acc_touched to avoid double removal confusion
                         if c in acc_touched_left: acc_touched_left.remove(c)
                         if c in acc_touched_right: acc_touched_right.remove(c)

                # Check for task completion & Remove objects
                if scheduler:
                    # Get current states to identify Base/Free arm for Phase 2
                    cl_st, cl_inf = scheduler.get_arm_state(t + 1, 'left')
                    cr_st, cr_inf = scheduler.get_arm_state(t + 1, 'right')

                    # Left
                    ended_task_l = scheduler.get_task_ending_at(t + 1, 'left')
                    if ended_task_l:
                        print(f"DEBUG: Task Ended for Left at step {t}: {ended_task_l['info']}")
                        if 'Phase 1' not in ended_task_l['info']:
                            grasped = get_grasped_cubes(env.physics)
                            nearby = get_proximity_cubes(env.physics)
                            currently_touching = get_touched_cubes_per_arm(env.physics)
                            
                            # Protection: Hold, Touch, or Proximity
                            protected_any = (grasped['left'] | grasped['right'] | 
                                             currently_touching['left'] | currently_touching['right'] | 
                                             nearby)
                            
                            to_remove = acc_touched_left - protected_any
                            
                            if to_remove:
                                print(f"[Step {t}] Left Task Ended. Removing objects: {to_remove}")
                                remove_cubes(env.physics, list(to_remove))
                            else:
                                print(f"[Step {t}] Left Task Ended. No objects to remove (Protected: {acc_touched_left & protected_any})")
                                
                            # Preserve protected items for future removal
                            acc_touched_left = acc_touched_left & protected_any
                        else:
                            # Phase 1 Ended. Transfer objects to Base arm if this is Free arm.
                            if 'Phase 2 (Place-Base)' in cl_inf:
                                print(f"[Step {t}] Left Phase 1 Ended. Legally persisting {acc_touched_left} (Base Arm).")
                            else:
                                print(f"[Step {t}] Left Phase 1 Ended. Transferring {acc_touched_left} to Right (Base Arm).")
                                acc_touched_right.update(acc_touched_left)
                                acc_touched_left.clear()
                    
                    # Right
                    ended_task_r = scheduler.get_task_ending_at(t + 1, 'right')
                    if ended_task_r:
                        print(f"DEBUG: Task Ended for Right at step {t}: {ended_task_r['info']}")
                        if 'Phase 1' not in ended_task_r['info']:
                            grasped = get_grasped_cubes(env.physics)
                            nearby = get_proximity_cubes(env.physics)
                            currently_touching = get_touched_cubes_per_arm(env.physics)

                            protected_any = (grasped['left'] | grasped['right'] | 
                                             currently_touching['left'] | currently_touching['right'] | 
                                             nearby)
                            
                            to_remove = acc_touched_right - protected_any

                            if to_remove:
                                print(f"[Step {t}] Right Task Ended. Removing objects: {to_remove}")
                                remove_cubes(env.physics, list(to_remove))
                            else:
                                print(f"[Step {t}] Right Task Ended. No objects to remove (Protected: {acc_touched_right & protected_any})")
                                
                            acc_touched_right = acc_touched_right & protected_any
                        else:
                            # Phase 1 Ended. Transfer objects to Base arm if this is Free arm.
                            if 'Phase 2 (Place-Base)' in cr_inf:
                                print(f"[Step {t}] Right Phase 1 Ended. Legally persisting {acc_touched_right} (Base Arm).")
                            else:
                                print(f"[Step {t}] Right Phase 1 Ended. Transferring {acc_touched_right} to Left (Base Arm).")
                                acc_touched_left.update(acc_touched_right)
                                acc_touched_right.clear()
                
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
                 video_dir = args.save_video if isinstance(args.save_video, str) else 'videos'
                 if not os.path.exists(video_dir):
                     os.makedirs(video_dir)
                     
                 status_str = "success" if is_success else "fail"
                 video_path = os.path.join(video_dir, f'eval_ep{episode_count}_{status_str}_r{current_reward}.mp4')
                 
                 # Detect shape from first frame
                 h, w, _ = video_frames[0].shape
                 fps = 30
                 out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                 for frame in video_frames:
                     frame_bgr = frame[:, :, [2, 1, 0]]
                     out.write(frame_bgr)
                 out.release()
                 print(f"Saved video to {video_path}")
             
            # Save Stats to CSV
            if args.save_stats_path:
                # Ensure directory exists with validation
                stats_dir = os.path.dirname(args.save_stats_path)
                if stats_dir and not os.path.exists(stats_dir):
                    try:
                        os.makedirs(stats_dir, exist_ok=True)
                        print(f"Created directory: {stats_dir}")
                    except OSError as e:
                        print(f"Error creating directory {stats_dir}: {e}")

                # Get Task Status
                try:
                    task = env._task
                    col_seq = task.color_sequence if task.color_sequence else ['?']*10
                    
                    # Prepare Row Data
                    row = {
                        'Episode': episode_count,
                        'Total_Reward': current_reward,
                        'Is_Success': is_success
                    }
                    
                    # Indices
                    indep_set = getattr(task, 'completed_independent_cubes', set())
                    coop_set = getattr(task, 'completed_cooperative_pairs', set())
                    
                    # Flatten coop set for easy lookup
                    coop_indices = set()
                    for (g, b) in coop_set:
                        coop_indices.add(g)
                        coop_indices.add(b)
                        
                    for i in range(10):
                        c_code = col_seq[i] if i < len(col_seq) else '?'
                        status = "Fail"
                        reward = 0
                        
                        if i in indep_set:
                            status = "Indep_Success"
                            reward = 1
                        elif i in coop_indices:
                            status = "Coop_Success"
                            reward = 2
                        
                        row[f'Obj{i}_Color'] = c_code
                        row[f'Obj{i}_Status'] = status
                        row[f'Obj{i}_Reward'] = reward
                        
                    # Write to CSV
                    file_exists = os.path.isfile(args.save_stats_path)
                    fieldnames = ['Episode', 'Total_Reward', 'Is_Success']
                    for i in range(10):
                        fieldnames.extend([f'Obj{i}_Color', f'Obj{i}_Status', f'Obj{i}_Reward'])
                        
                    with open(args.save_stats_path, 'a', newline='') as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        if not file_exists:
                            writer.writeheader()
                        writer.writerow(row)
                    print(f"Saved stats to {args.save_stats_path}")
                    
                except Exception as e:
                    print(f"Error saving stats: {e}")

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
    
    parser.add_argument('--commands', action='store', type=str, help='Command sequence (e.g. ICI)', required=False)
    parser.add_argument('--color_sequence', action='store', type=str, help='Color sequence', default=None)
    parser.add_argument('--sequence_file', action='store', type=str, help='Path to CSV sequence file', default=None)
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
    parser.add_argument('--save_video', nargs='?', const='videos', type=str, help='Save execution video (optional path, default "videos")')
    parser.add_argument('--save_stats_path', action='store', type=str, help='Path to save episode statistics CSV')
    parser.add_argument('--num_rollouts', action='store', type=int, default=1, help='Number of evaluation episodes')
    parser.add_argument('--episode_len', action='store', type=int, default=None, help='Override episode length')
    parser.add_argument('--max_timesteps', action='store', type=int, default=None, help='Hard limit on episode steps')
    parser.add_argument('--sync_arms', action='store_true', help='Enable Sync Wait logic (Hold) before Cooperative tasks')
    parser.add_argument('--reset_on_subtask', action='store_true', help='Reset independent policy buffers on subtask switch')
    parser.add_argument('--x_shift', action='store', type=float, default=0.0, help='Shift all objects along X-axis')
    parser.add_argument('--x_shift_start_idx', action='store', type=int, default=0, help='Start index for applying x-shift (0-indexed)')
    
    args = parser.parse_args()
    main(args)
