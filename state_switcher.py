
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
from collections import defaultdict
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
from piper_constants import PUPPET_GRIPPER_POSITION_NORMALIZE_FN
from utils import load_data
from utils import sample_redbox_pose, sample_insertion_pose, sample_greenbox_pose, sample_bluebox_pose
from utils import compute_dict_mean, set_seed, detach_dict
from policy import ACTPolicy
from interactive_policy import InteractivePolicy

# Import Sim Env
from piper_sim_env import REDBOX_POSE, GREENBOX_POSE, BLUEBOX_POSE, MANYCUBES_COLORS, MANYCUBES_POSES
from piper_sim_env import make_sim_env
from piper_ee_sim_env import make_ee_sim_env
from utils import apply_rgb_mask_to_strip, apply_rgb_mask_to_right_strip, apply_policy_mask


def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy

# --- HITL UTILS ---
import termios
import tty

def get_hitl_key():
    """Read a single keypress from stdin without waiting for Enter (Non-blocking check).
    Assumes terminal is already in cbreak mode (set at startup)."""
    if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
        ch = sys.stdin.read(1)
        return ch
    return None

def save_hitl_data(save_dir, episode_idx, images_np, labels_l, labels_r):
    """Save HITL data to HDF5."""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    path = os.path.join(save_dir, f'episode_{episode_idx}.hdf5')
    with h5py.File(path, 'w') as f:
        # Observations
        f.create_dataset('observations/images/top', data=images_np, compression='gzip')
        
        # HITL Labels (Frame-wise)
        f.create_dataset('labels/left', data=labels_l)
        f.create_dataset('labels/right', data=labels_r)
        
        # Metadata flag
        f.create_group('metadata')
        f['metadata'].attrs['hitl'] = True
        
    print(f"Saved HITL data to {path}")

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
            
            # 2. Set Alpha to 0 (Visual Hide)
            name = f'cube_{i}'
            geom_id = physics.model.name2id(name, 'geom')
            physics.model.geom_rgba[geom_id, 3] = 0.0
        except Exception as e:
            pass
    physics.forward()

def sync_envs(main_physics, shadow_physics):
    """Sync robot/object state AND visual properties (geom_rgba) from main to shadow."""
    # 1. State Sync
    shadow_physics.data.qpos[:] = main_physics.data.qpos[:]
    shadow_physics.data.qvel[:] = main_physics.data.qvel[:]
    
    # 2. Visual Sync (geom_rgba determines cube colors)
    shadow_physics.model.geom_rgba[:] = main_physics.model.geom_rgba[:]
    
    shadow_physics.forward()

def get_spatial_object_map(physics):
    """Map L0, R0 etc to cube indices based on x-coordinate relative to x=0."""
    cubes = []
    for i in range(10):
        try:
            name = f'cube_{i}'
            bid = physics.model.name2id(name, 'body')
            x = physics.data.xpos[bid][0]
            cubes.append({'id': i, 'x': x})
        except Exception:
            pass
    
    # Sort symmetrically (Matches extract_transitions.py)
    # L0 is closest to center (highest X), L1 is further left.
    left_cubes = sorted([c for c in cubes if c['x'] < 0], key=lambda c: c['x'], reverse=True)
    # R0 is closest to center (lowest X), R1 is further right.
    right_cubes = sorted([c for c in cubes if c['x'] >= 0], key=lambda c: c['x'])
    
    obj_map = {}
    for i, c in enumerate(left_cubes): obj_map[f'L{i}'] = c['id']
    for i, c in enumerate(right_cubes): obj_map[f'R{i}'] = c['id']
    
    return obj_map

def mode_to_char(mode_str):
    if mode_str == 'COOP':
        return 'C'
    if mode_str == 'HOLD':
        return 'H'
    return 'I'

def compose_video_frame(main_rgb, coop_rgb, indep_l_rgb, indep_r_rgb, left_mode_char, right_mode_char):
    tile_w, tile_h = 640, 360
    main_tile = cv2.resize(main_rgb, (tile_w, tile_h), interpolation=cv2.INTER_LINEAR)
    coop_tile = cv2.resize(coop_rgb, (tile_w, tile_h), interpolation=cv2.INTER_LINEAR)
    indep_l_tile = cv2.resize(indep_l_rgb, (tile_w, tile_h), interpolation=cv2.INTER_LINEAR)
    indep_r_tile = cv2.resize(indep_r_rgb, (tile_w, tile_h), interpolation=cv2.INTER_LINEAR)

    top_row = np.hstack([main_tile, coop_tile])
    bottom_row = np.hstack([indep_l_tile, indep_r_tile])
    canvas = np.vstack([top_row, bottom_row])

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, f"{left_mode_char}", (20, canvas.shape[0] - 20), font, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
    text_r = f"{right_mode_char}"
    text_size, _ = cv2.getTextSize(text_r, font, 1.1, 2)
    cv2.putText(canvas, text_r, (canvas.shape[1] - text_size[0] - 20, canvas.shape[0] - 20), font, 1.1, (255, 255, 255), 2, cv2.LINE_AA)

    return canvas

def get_heuristic_targets(physics, spatial_map, color_sequence):
    """
    Determine target indices based on mode requirements and spatial heuristics.
    Returns: (target_indices_i, target_indices_c)
    """
    target_indices_i = []
    target_indices_c = []
    
    def safe_get_color(idx):
        try: return color_sequence[idx]
        except Exception:
            return None

    def is_removed_cube(idx):
        try:
            geom_id = physics.model.name2id(f'cube_{idx}', 'geom')
            alpha = float(physics.model.geom_rgba[geom_id, 3])
        except Exception:
            alpha = 1.0
        try:
            body_id = physics.model.name2id(f'cube_{idx}', 'body')
            z_pos = float(physics.data.xpos[body_id][2])
        except Exception:
            z_pos = 0.0
        return (alpha <= 0.01) or (z_pos < -1.0)

    # 1. Cooperative Shadow (target_indices_c)
    coop_candidates = []
    for label, c_idx in spatial_map.items():
        try:
            if is_removed_cube(c_idx):
                continue
            bid = physics.model.name2id(f'cube_{c_idx}', 'body')
            x_pos = physics.data.xpos[bid][0]
            # Ensure STRICTLY no red objects ('r') enter here
            color = safe_get_color(c_idx)
            if x_pos < 0.3:
                if color in ['g', 'b']:
                    coop_candidates.append((x_pos, c_idx, color))
        except Exception:
            pass
    
    if coop_candidates:
        rightmost_g = max([c for c in coop_candidates if c[2] == 'g'], key=lambda x: x[0], default=None)
        rightmost_b = max([c for c in coop_candidates if c[2] == 'b'], key=lambda x: x[0], default=None)
        # Cooperative view must be updated as a G/B set (no partial single-color display).
        if rightmost_g is not None and rightmost_b is not None:
            target_indices_c.append(rightmost_g[1])
            target_indices_c.append(rightmost_b[1])

    # 2. Independent Shadow (target_indices_i)
    # Keep this arm-agnostic here: collect all red cubes.
    # Per-arm selection (x=0 split, max-X on each side) is done later.
    red_candidates = []
    for label, c_idx in spatial_map.items():
        try:
            if is_removed_cube(c_idx):
                continue
            color = safe_get_color(c_idx)
            if color == 'r':
                red_candidates.append(c_idx)
        except:
            pass

    target_indices_i = red_candidates

    return list(set(target_indices_i)), list(set(target_indices_c))

def hide_objects(physics, target_indices, env_name="?"):
    """Hide objects NOT in target_indices AND not already removed."""
    target_set = set(target_indices)
    for i in range(10):
        name = f'cube_{i}'
        
        # 1. Start with Geom visibility
        try:
            geom_id = physics.model.name2id(name, 'geom')
            if i in target_set:
                physics.model.geom_rgba[geom_id, 3] = 1.0 # Show
            else:
                physics.model.geom_rgba[geom_id, 3] = 0.0 # Hide
        except Exception:
            pass
        
        # 2. Move qpos if hiding (independent of geom success)
        if i not in target_set:
            try:
                joint_name = f'cube_{i}_joint'
                addr = physics.model.name2id(joint_name, 'joint')
                qpos_adr = physics.model.jnt_qposadr[addr]
                physics.data.qpos[qpos_adr + 2] = -5.0 # Well below table
            except Exception:
                pass
            
    physics.forward()

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
                    except Exception:
                        pass
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
                    except Exception:
                        pass
                
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
        except Exception:
            pass
    
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
        except Exception:
            pass
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
                except Exception:
                    pass

    # 3. Propagate (Transitive Closure) for stacked objects
    changed = True
    while changed:
        changed = False
        for n1, n2 in contacts:
            c1_idx = -1
            c2_idx = -1
            
            if n1.startswith('cube_'):
                try: c1_idx = int(n1.split('_')[1])
                except Exception:
                    pass
            
            if n2.startswith('cube_'):
                try: c2_idx = int(n2.split('_')[1])
                except Exception:
                    pass
                
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
    return get_cubes_touching_targets(physics, {'cushion', 'cushion1', 'cushion2'})

def is_arm_at_home(current_qpos, home_qpos, threshold=0.1):
    """Check if a single arm (7-dim) is close to its home pose."""
    diff = np.abs(current_qpos - home_qpos)
    max_diff = np.max(diff)
    return max_diff < threshold


def is_at_home(current_qpos, threshold=0.1, gripper_threshold=0.8, return_details=False):
    """Hierarchical-style home check on full 14-dim qpos.

    - Arm joints must be close to START_ARM_POSE
    - Both grippers must be open enough
    """
    home_l = np.array(START_ARM_POSE[:6])
    home_r = np.array(START_ARM_POSE[8:14])

    diff_l = np.max(np.abs(current_qpos[:6] - home_l))
    diff_r = np.max(np.abs(current_qpos[7:13] - home_r))
    arms_at_home = (diff_l < threshold) and (diff_r < threshold)

    left_open = current_qpos[6] > gripper_threshold
    right_open = current_qpos[13] > gripper_threshold
    grippers_open = left_open and right_open

    home_ok = arms_at_home and grippers_open
    if return_details:
        return home_ok, diff_l, diff_r, left_open, right_open
    return home_ok

def quaternion_multiply(q1, q2):
    """Multiply two quaternions. q1 * q2"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def quaternion_inverse(q):
    """Inverse of quaternion [w, x, y, z]"""
    return np.array([q[0], -q[1], -q[2], -q[3]]) / np.dot(q, q)

def rotate_vector_by_quaternion(v, q):
    """Rotate vector v by quaternion q."""
    # p = [0, v]
    # p' = q * p * q_inv
    q_vec = np.array([0, v[0], v[1], v[2]])
    q_inv = quaternion_inverse(q)
    temp = quaternion_multiply(q, q_vec)
    result = quaternion_multiply(temp, q_inv)
    return result[1:] # Return vector part

def apply_magnet_logic(physics, magnetized_pairs, color_sequence=None):
    """
    Visual-Only Stacking Logic (Magnet).
    If a Green cube is close to a Blue cube, disable Green's collision and 
    lock its relative transform to the Blue cube.
    """
    # 1. Identify Green and Blue cubes
    if color_sequence:
        greens = [i for i, c in enumerate(color_sequence) if c == 'g']
        blues = [i for i, c in enumerate(color_sequence) if c == 'b']
    else:
        # Fallback for sim_many_cubes default (8=Green, 9=Blue)
        greens = [8]
        blues = [9]

    # 2. Check for new magnetizations
    # Trigger only when:
    # - Green is above Blue in Z
    # - Z distance (G over B) <= 0.6
    # - XY distance <= 0.4
    # - both cubes are lifted (z > 0.05)
    z_gap_max = 0.06
    xy_dist_max = 0.04
    min_z_height = 0.05
    
    for g_idx in greens:
        if g_idx in magnetized_pairs: continue
        
        try:
            g_body_id = physics.model.name2id(f'cube_{g_idx}', 'body')
            g_pos = physics.data.xpos[g_body_id].copy()
            g_quat = physics.data.xquat[g_body_id].copy() # [w, x, y, z]
            
            for b_idx in blues:
                 b_body_id = physics.model.name2id(f'cube_{b_idx}', 'body')
                 b_pos = physics.data.xpos[b_body_id].copy()
                 b_quat = physics.data.xquat[b_body_id].copy() 
                 
                 z_gap = g_pos[2] - b_pos[2]
                 xy_dist = np.linalg.norm(g_pos[:2] - b_pos[:2])
                 both_high_enough = (g_pos[2] > min_z_height) and (b_pos[2] > min_z_height)
                 green_above_blue = z_gap >= 0.0
                 z_gap_ok = z_gap <= z_gap_max
                 xy_ok = xy_dist <= xy_dist_max

                 if both_high_enough and green_above_blue and z_gap_ok and xy_ok:
                     print(
                         f"Magnet Triggered: Green {g_idx} -> Blue {b_idx} "
                         f"(z_gap={z_gap:.4f}, xy={xy_dist:.4f}, gz={g_pos[2]:.4f}, bz={b_pos[2]:.4f})"
                     )
                     
                     # Calculate Relative Transform (Offset)
                     # Rel Pos: Vector from Blue to Green, in Blue's local frame?
                     # OR just Global Offset? User asked to "preserve relative position".
                     # If we just store global offset (g - b), it won't rotate with Blue.
                     # We need Local Offset: v_local = rotate(v_global, q_inv)
                     # Then v_global_new = rotate(v_local, q_new)
                     
                     # 1. Calculate Relative Position in Blue's Frame
                     global_offset = g_pos - b_pos
                     b_quat_inv = quaternion_inverse(b_quat)
                     rel_pos = rotate_vector_by_quaternion(global_offset, b_quat_inv)
                     
                     # 2. Calculate Relative Rotation
                     # q_rel = q_blue_inv * q_green
                     rel_quat = quaternion_multiply(b_quat_inv, g_quat)
                     
                     magnetized_pairs[g_idx] = {
                         'blue_idx': b_idx,
                         'rel_pos': rel_pos,
                         'rel_quat': rel_quat
                     }
                     
                     # Disable Collision for Green
                     # Note: modifying model.geom_contype is permanent for the session
                     g_geom_id = physics.model.name2id(f'cube_{g_idx}', 'geom')
                     physics.model.geom_contype[g_geom_id] = 0
                     physics.model.geom_conaffinity[g_geom_id] = 0
                     
                     break # Only magnetize to one
        except Exception as e:
            print(f"Magnet check error: {e}")
            pass

    # 3. Apply updates for magnetized pairs
    for g_idx, data in magnetized_pairs.items():
        try:
            b_idx = data['blue_idx']
            rel_pos = data['rel_pos']
            rel_quat = data['rel_quat']
            
            b_body_id = physics.model.name2id(f'cube_{b_idx}', 'body')
            b_pos = physics.data.xpos[b_body_id].copy()
            b_quat = physics.data.xquat[b_body_id].copy()
            
            # Calculate new Green Pose
            # Global Offset = rotate(rel_pos, b_quat)
            global_offset = rotate_vector_by_quaternion(rel_pos, b_quat)
            target_pos = b_pos + global_offset
            
            # Target Quat = b_quat * rel_quat
            target_quat = quaternion_multiply(b_quat, rel_quat)
            
            # Apply to Joint (Teleport)
            start_idx = physics.model.name2id(f'cube_{g_idx}_joint', 'joint')
            qpos_adr = physics.model.jnt_qposadr[start_idx]
            
            physics.data.qpos[qpos_adr : qpos_adr+3] = target_pos
            physics.data.qpos[qpos_adr+3 : qpos_adr+7] = target_quat

            # Zero out velocity to prevent gravity fight
            qvel_adr = physics.model.jnt_dofadr[start_idx]
            physics.data.qvel[qvel_adr : qvel_adr+6] = 0.0
            
        except Exception as e:
            print(f"Magnet apply error for G{g_idx}: {e}")
            pass

def reset_magnet_logic(physics, color_sequence=None):
    """Restore collision for ALL cubes at reset."""
    # Reset all 10 cubes regardless of color sequence to be safe
    for i in range(10):
        try:
             geom_id = physics.model.name2id(f'cube_{i}', 'geom')
             physics.model.geom_contype[geom_id] = 1
             physics.model.geom_conaffinity[geom_id] = 1
        except Exception:
            pass


def get_image_dual(ts, camera_names, mask=False):
    curr_images = []
    for cam_name in camera_names:
        # Sim Env render (H, W, C)
        curr_image_np = ts.observation['images'][cam_name].copy()
        
        # Apply Policy Mask (Mask Red for COOP)
        if mask:
            curr_image_np = apply_policy_mask(curr_image_np, 'COOP')

        # Convert to Tensor (1, C, H, W)
        curr_image_t = torch.from_numpy(curr_image_np).permute(2, 0, 1).unsqueeze(0).float().cuda() / 255.0
        curr_images.append(curr_image_t.squeeze(0))
    curr_image = torch.stack(curr_images, dim=0).unsqueeze(0) # (1, num_cam, C, H, W)
    return curr_image

def get_image_independent(ts, camera_names, arm, mask=False):
    curr_images = []
    for cam_name in camera_names:
        # Sim Env render (H, W, C)
        curr_image_np = ts.observation['images'][cam_name].copy()
        
        # Apply Policy Mask (Mask Green/Blue for INDEP)
        if mask:
             curr_image_np = apply_policy_mask(curr_image_np, 'INDEP')

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
    
    # Handle new checkpoint format (nested with model_state_dict and optimizer_state_dict)
    if 'model_state_dict' in loaded_state_dict:
        print(f"Detected new checkpoint format, extracting model_state_dict...")
        loaded_state_dict = loaded_state_dict['model_state_dict']
    
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
    set_seed(args.seed)
    scripted_ll_override_all_mode = bool(getattr(args, 'debug_scripted_low_level', False) or getattr(args, 'unit_test_scripted_low_level', False))
    scripted_ll_coop_only_mode = bool(getattr(args, 'debug_scripted_coop_low_level_only', False))
    scripted_ll_test_mode = bool(scripted_ll_override_all_mode or scripted_ll_coop_only_mode)
    
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

    # High-level oracle metadata (optional): replay pair_at_t0/transitions_json as HL states
    hl_oracle_by_seqrow = defaultdict(list)
    hl_oracle_by_episode = {}
    hl_oracle_cursor_by_seqrow = defaultdict(int)
    hl_oracle_rows = []
    hl_oracle_episode_order = None

    def _parse_int_safe(v, default=-1):
        try:
            return int(v)
        except Exception:
            return default

    def _mode_char_to_state(ch):
        ch = (ch or 'H').upper()
        if ch == 'C':
            return STATE_COOP
        if ch == 'I':
            return STATE_INDEP
        return STATE_HOLD

    def _pair_to_states(pair):
        p = (pair or 'HH').strip().upper()
        if len(p) < 2:
            p = (p + 'HH')[:2]
        return _mode_char_to_state(p[0]), _mode_char_to_state(p[1])

    def _is_valid_color_seq(s):
        s = (s or '').strip().lower()
        return len(s) == 10 and all(c in ['r', 'g', 'b'] for c in s)

    def _oracle_states_at_t(schedule, step_t):
        pair = schedule['pair0']
        for tr in schedule['transitions']:
            if step_t >= tr['t']:
                pair = tr['pair']
            else:
                break
        return _pair_to_states(pair), pair

    def _select_oracle_schedule(ep_idx, sequence_row_idx):
        if hl_oracle_episode_order is not None and 0 <= ep_idx < len(hl_oracle_episode_order):
            row_idx = hl_oracle_episode_order[ep_idx]
            if 0 <= row_idx < len(hl_oracle_rows):
                return hl_oracle_rows[row_idx]
        if sequence_row_idx is not None and sequence_row_idx in hl_oracle_by_seqrow:
            rows = hl_oracle_by_seqrow[sequence_row_idx]
            if rows:
                cursor = hl_oracle_cursor_by_seqrow[sequence_row_idx]
                picked = rows[cursor % len(rows)]
                hl_oracle_cursor_by_seqrow[sequence_row_idx] = cursor + 1
                return picked
        if ep_idx in hl_oracle_by_episode:
            return hl_oracle_by_episode[ep_idx]
        return None

    if args.hl_oracle_metadata_csv:
        if not os.path.exists(args.hl_oracle_metadata_csv):
            print(f"Error: hl_oracle_metadata_csv not found: {args.hl_oracle_metadata_csv}")
            return
        print(f"Loading HL oracle metadata: {args.hl_oracle_metadata_csv}")
        loaded_oracle_rows = 0
        with open(args.hl_oracle_metadata_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                pair0 = (row.get('pair_at_t0') or 'HH').strip().upper()
                trans_raw = row.get('transitions_json') or '[]'
                try:
                    transitions = json.loads(trans_raw)
                except Exception:
                    transitions = []

                parsed_transitions = []
                for tr in transitions:
                    try:
                        tr_t = int(tr.get('t', 0))
                        tr_pair = str(tr.get('pair', 'HH')).strip().upper()
                        if len(tr_pair) >= 2:
                            parsed_transitions.append({'t': tr_t, 'pair': tr_pair[:2]})
                    except Exception:
                        continue
                parsed_transitions.sort(key=lambda x: x['t'])

                entry = {
                    'pair0': pair0[:2] if len(pair0) >= 2 else 'HH',
                    'transitions': parsed_transitions,
                    'color_sequence': (row.get('color_sequence') or '').strip().lower(),
                    'command_sequence': (row.get('command_sequence') or '').strip().upper(),
                }

                seq_row = _parse_int_safe(row.get('sequence_row'), default=-1)
                ep = _parse_int_safe(row.get('episode'), default=-1)
                entry['sequence_row'] = seq_row if seq_row >= 0 else None
                if seq_row >= 0:
                    hl_oracle_by_seqrow[seq_row].append(entry)
                if ep >= 0:
                    hl_oracle_by_episode[ep] = entry
                hl_oracle_rows.append(entry)
                loaded_oracle_rows += 1
        print(f"Loaded HL oracle rows: {loaded_oracle_rows}")
        if len(hl_oracle_rows) == 0:
            print("Error: No valid rows found in hl_oracle_metadata_csv.")
            return

        # Always randomize episode execution order when oracle CSV is used.
        hl_oracle_episode_order = np.random.permutation(len(hl_oracle_rows)).tolist()
        print("[Switcher] ORACLE row order randomized for this run.")

        if args.num_rollouts > len(hl_oracle_rows):
            print(f"Warning: num_rollouts ({args.num_rollouts}) > oracle rows ({len(hl_oracle_rows)}). Clamping to {len(hl_oracle_rows)}.")
            args.num_rollouts = len(hl_oracle_rows)
    
    # Handle Sequence Loading
    sequences = []
    sequence_commands = []
    sequence_episode_order = None
    if args.sequence_file and not args.hl_oracle_metadata_csv:
         if os.path.exists(args.sequence_file):
             print(f"Loading sequence file: {args.sequence_file}")
             with open(args.sequence_file, 'r') as f:
                 reader = csv.reader(f)
                 for row in reader:
                     if not row:
                         continue

                     # Preferred format: col0=color_sequence, col1=commands, col2=status(optional)
                     # Fallback support: col1=color_sequence in older/custom CSV layouts.
                     candidate0 = row[0].strip() if len(row) >= 1 else ""
                     candidate1 = row[1].strip() if len(row) >= 2 else ""

                     if _is_valid_color_seq(candidate0):
                         sequences.append(candidate0.lower())
                         sequence_commands.append(candidate1.upper() if len(candidate1) > 0 else None)
                     elif _is_valid_color_seq(candidate1):
                         sequences.append(candidate1.lower())
                         sequence_commands.append(candidate0.upper() if len(candidate0) > 0 else None)
                     else:
                         # Skip header/invalid rows silently.
                         continue
             print(f"Loaded {len(sequences)} sequences.")
             if len(sequences) == 0:
                 print(f"Error: No valid color sequences found in {args.sequence_file}.")
                 return
             if args.num_rollouts > len(sequences):
                 print(f"Warning: num_rollouts ({args.num_rollouts}) > loaded sequences ({len(sequences)}). Clamping num_rollouts to {len(sequences)}.")
                 args.num_rollouts = len(sequences)

             # Randomized episode-to-sequence mapping (no top-down fixed order)
             sequence_episode_order = np.random.permutation(len(sequences)).tolist()
             print("[Switcher] Sequence order randomized for this run.")
         else:
             print(f"Error: Sequence file {args.sequence_file} not found.")
             return
    elif args.sequence_file and args.hl_oracle_metadata_csv:
         print("[Switcher] hl_oracle_metadata_csv is set. Ignoring --sequence_file and using oracle metadata colors/commands.")

    # Handle Color Sequence (Default or Override)
    default_color_seq = None
    if args.color_sequence:
        default_color_seq = args.color_sequence.lower()
        if len(default_color_seq) != 10:
             print(f"WARNING: Color sequence must be 10 chars. Got {len(default_color_seq)}. Using default.")
             default_color_seq = None
        elif any(c not in ['r', 'g', 'b'] for c in default_color_seq):
             print(f"WARNING: Invalid chars in color sequence. Using default.")
             default_color_seq = None
             
    if default_color_seq:
         print(f"Setting Default Color Sequence: {default_color_seq}")
         MANYCUBES_COLORS[0] = list(default_color_seq)

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
    
    pre_process_left = lambda s_qpos: (s_qpos - stats_left['qpos_mean']) / stats_left['qpos_std']
    
    pre_process_right = lambda s_qpos: (s_qpos - stats_right['qpos_mean']) / stats_right['qpos_std']

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
    
    import piper_constants
    if args.x_shift:
        piper_constants.MANYCUBES_CONFIG['x_shift'] = args.x_shift
        piper_constants.MANYCUBES_CONFIG['x_shift_start_idx'] = args.x_shift_start_idx
        print(f"Applying x-shift: {args.x_shift} starting from index {args.x_shift_start_idx}")

    print(f"Creating Simulation Environment with time_limit={time_limit:.2f}s ({max_timesteps} steps + buffer)")
    env = make_sim_env(task_name, camera_names=camera_names, time_limit=time_limit)
    env_shadow_c = make_sim_env(task_name, camera_names=camera_names, time_limit=time_limit)
    env_shadow_i_left = make_sim_env(task_name, camera_names=camera_names, time_limit=time_limit)
    env_shadow_i_right = make_sim_env(task_name, camera_names=camera_names, time_limit=time_limit)
    
    if onscreen_render:
         plt.ion()
         fig, (ax_main, ax_c, ax_i_l, ax_i_r) = plt.subplots(1, 4, figsize=(20, 5))
         plt_img_main = ax_main.imshow(np.zeros((240, 320, 3), dtype=np.uint8))
         plt_img_c = ax_c.imshow(np.zeros((240, 320, 3), dtype=np.uint8))
         plt_img_i_l = ax_i_l.imshow(np.zeros((240, 320, 3), dtype=np.uint8))
         plt_img_i_r = ax_i_r.imshow(np.zeros((240, 320, 3), dtype=np.uint8))
         ax_main.set_title("Main")
         ax_c.set_title("Coop Shadow")
         ax_i_l.set_title("Indep Shadow (Left)")
         ax_i_r.set_title("Indep Shadow (Right)")
    
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
            # Explicitly re-sample poses for ManyCubes task to ensure randomization
            # ManyCubesTask uses GREENBOX_POSE/BLUEBOX_POSE during initialization
            GREENBOX_POSE[0] = sample_greenbox_pose()
            BLUEBOX_POSE[0] = sample_bluebox_pose()
            pass
        
        ts = env.reset()
        # Explicitly reset shadow environments to ensure they pick up new config (e.g. colors)
        env_shadow_c.reset()
        env_shadow_i_left.reset()
        env_shadow_i_right.reset()
        return ts

    def setup_debug_scripted_low_level(ts_main, command_seq):
        """Create EE scripted runner aligned to current main episode initialization."""
        if not scripted_ll_test_mode:
            return None, None, None

        if 'sim_many_cubes' in task_name:
            env_state = np.array(ts_main.observation['env_state']).copy()
            if env_state.shape[0] >= 70:
                poses = {}
                for i in range(10):
                    poses[i] = env_state[i*7:(i+1)*7].copy()
                MANYCUBES_POSES[0] = poses

        ee_env = make_ee_sim_env(task_name)
        ts_ee = ee_env.reset()

        scripted_cmds = command_seq
        if scripted_cmds is None or len(scripted_cmds) == 0:
            # Fallback command expansion from color sequence (debug only)
            seq = args.color_sequence if args.color_sequence else ""
            r_n = seq.count('r')
            c_n = min(seq.count('g'), seq.count('b'))
            scripted_cmds = ('I' * r_n) + ('C' * c_n)

        scripted_policy = InteractivePolicy(inject_noise=False, color_sequence=list(args.color_sequence))
        scripted_policy.init_pose(ts_ee)
        for cmd in scripted_cmds:
            if cmd in ['I', 'C', 'L', 'R', 'T', 'B']:
                scripted_policy.schedule_command(cmd, ts_ee)

        print(f"[Debug Scripted LL] Enabled. Commands={scripted_cmds}")
        return ee_env, scripted_policy, ts_ee

    def sync_debug_ee_objects_from_main(ts_main, ee_env):
        """Sync cube poses from main env observation to EE debug env every step."""
        if ee_env is None or ts_main is None:
            return
        try:
            env_state = np.array(ts_main.observation.get('env_state', []))
            if env_state.shape[0] < 70:
                return
            for i in range(10):
                try:
                    joint_id = ee_env.physics.model.name2id(f'cube_{i}_joint', 'joint')
                    qpos_adr = ee_env.physics.model.jnt_qposadr[joint_id]
                    ee_env.physics.data.qpos[qpos_adr:qpos_adr+7] = env_state[i*7:(i+1)*7]
                except Exception:
                    continue
            ee_env.physics.forward()
        except Exception:
            pass

    def apply_next_episode_sequence(ep_idx):
        if args.hl_oracle_metadata_csv:
            oracle_entry = _select_oracle_schedule(ep_idx, None)
            if oracle_entry is None:
                print(f"[Switcher] Error: No oracle metadata row for episode {ep_idx}.")
                return False

            oracle_color = oracle_entry.get('color_sequence', '')
            if _is_valid_color_seq(oracle_color):
                seq_list = list(oracle_color)
                MANYCUBES_COLORS[0] = seq_list
                args.color_sequence = ''.join(seq_list)
            else:
                print(f"[Switcher] Error: Oracle row for episode {ep_idx} has invalid/missing color_sequence: '{oracle_color}'")
                return False

            args.current_command_sequence = oracle_entry.get('command_sequence') or None
            args.current_sequence_row = oracle_entry.get('sequence_row')
            print(f"[Switcher] Applied ORACLE sequence for Episode {ep_idx}: {args.color_sequence} (seq_row={args.current_sequence_row})")
            return True

        if not args.sequence_file:
            args.current_command_sequence = None
            args.current_sequence_row = None
            return True
        if ep_idx >= len(sequences):
            print(f"[Switcher] Error: Episode {ep_idx} exceeds loaded sequences ({len(sequences)}).")
            return False

        seq_idx = ep_idx
        if sequence_episode_order is not None and ep_idx < len(sequence_episode_order):
            seq_idx = sequence_episode_order[ep_idx]

        seq = sequences[seq_idx]
        cmd_seq = sequence_commands[seq_idx] if seq_idx < len(sequence_commands) else None
        seq_list = list(seq.lower().strip())
        if len(seq_list) == 10 and all(c in ['r', 'g', 'b'] for c in seq_list):
            MANYCUBES_COLORS[0] = seq_list
            # Update args so Magnet Logic sees it
            args.color_sequence = "".join(seq_list)
            args.current_command_sequence = cmd_seq
            args.current_sequence_row = seq_idx
            print(f"[Switcher] Applied Sequence for Episode {ep_idx} (row {seq_idx}): {args.color_sequence}")
            return True

        print(f"[Switcher] Error: Invalid sequence for Episode {ep_idx}: {seq}")
        return False

    # Set initial sequence (Episode 0)
    if not apply_next_episode_sequence(0):
        return

    ts = reset_with_new_pose()
    debug_ee_env, debug_scripted_policy, debug_ts_ee = setup_debug_scripted_low_level(ts, getattr(args, 'current_command_sequence', None))
    current_hl_oracle_schedule = _select_oracle_schedule(0, getattr(args, 'current_sequence_row', None)) if args.hl_oracle_metadata_csv else None
    if current_hl_oracle_schedule is not None:
        print(f"[HL Oracle] Enabled for episode 0 (sequence_row={getattr(args, 'current_sequence_row', None)})")
    
    old_settings = None
    if sys.stdin.isatty():
        old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        print("Ready. Press 'q' to quit.")
    else:
        print("Not a TTY, manual control disabled")
    
    print("\n\nReady!")
    print("Press 'q' to quit")
    
    # State tracking for chunk execution (per-policy to avoid cross-arm interference)
    step_in_chunk = 0 # Legacy/global tracker (kept for compatibility logs)
    step_in_chunk_dual = 0
    step_in_chunk_left = 0
    step_in_chunk_right = 0
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

    def clear_temporary_states(reset_transition_windows=False):
        nonlocal step_in_chunk_dual, step_in_chunk_left, step_in_chunk_right
        nonlocal current_action_chunk_dual, current_action_chunk_left, current_action_chunk_right
        nonlocal transition_window_active_l, transition_window_active_r

        all_time_actions_dual.fill_(float_nan)
        all_time_actions_left.fill_(float_nan)
        all_time_actions_right.fill_(float_nan)

        step_in_chunk_dual = 0
        step_in_chunk_left = 0
        step_in_chunk_right = 0

        current_action_chunk_dual = None
        current_action_chunk_left = None
        current_action_chunk_right = None

        if reset_transition_windows:
            transition_window_active_l = 0
            transition_window_active_r = 0

    # Home Pose
    home_pose = np.zeros(14)
    home_pose[:6] = START_ARM_POSE[:6]
    home_pose[6] = START_ARM_POSE[6]
    home_pose[7:13] = START_ARM_POSE[8:14]
    home_pose[13] = START_ARM_POSE[14]

    # State mapping for Labels
    def state_to_int(state_str):
        if state_str == 'HOLD': return STATE_HOLD
        if state_str == 'INDEP': return STATE_INDEP
        if state_str == 'COOP': return STATE_COOP
        return STATE_HOLD

    # HITL State Tracking (Initialize before loop)
    hitl_mode_l = 'AUTO'
    hitl_mode_r = 'AUTO'
    hitl_images = []
    hitl_labels_l = []
    hitl_labels_r = []

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
        ever_grasped_objects = set()
        removed_objects = set()
        coop_goal_pair_start_step = {}  # {(g_idx, b_idx): step_when_both_on_goal}
        coop_display_lock_pair = None   # (g_idx, b_idx) locked pair for coop shadow display
        active_coop_pair = None         # (g_idx, b_idx) currently assigned coop pair
        coop_pair_release_streak = {}   # {(g_idx, b_idx): consecutive non-contact steps after grasp}
        grasp_release_streak = {}       # {idx: consecutive frames released and unprotected}
        
        # Magnet Tracking
        reset_magnet_logic(env.physics, args.color_sequence)
        magnetized_pairs = {} # {g_idx: {'blue_idx': b, 'rel_pos': p, 'rel_quat': q}}
        
        # State Tracking Initialization
        prev_plan_l_state = None
        prev_plan_r_state = None
        last_active_l_state = None  # Track last non-HOLD state (INDEP or COOP)
        last_active_r_state = None
        state_history_l = collections.deque(maxlen=10)
        state_history_r = collections.deque(maxlen=10)
        hl_mode_l = STATE_INDEP
        hl_mode_r = STATE_INDEP
        last_hl_update_step_l = -1
        last_hl_update_step_r = -1
        
        # Delayed Stop Counters
        consecutive_at_home_l = 0
        consecutive_at_home_r = 0
        
        # Debounce State Tracking
        committed_plan_l_state = 'INDEP' # Start assumption
        committed_plan_r_state = 'INDEP' 
        if current_hl_oracle_schedule is not None:
            oracle_init_t = max(0, int(args.hl_oracle_future_steps))
            (oracle_l0, oracle_r0), _ = _oracle_states_at_t(current_hl_oracle_schedule, oracle_init_t)
            committed_plan_l_state = 'COOP' if oracle_l0 == STATE_COOP else ('HOLD' if oracle_l0 == STATE_HOLD else 'INDEP')
            committed_plan_r_state = 'COOP' if oracle_r0 == STATE_COOP else ('HOLD' if oracle_r0 == STATE_HOLD else 'INDEP')
        video_mode_l_char = mode_to_char(committed_plan_l_state)
        video_mode_r_char = mode_to_char(committed_plan_r_state)
        steps_since_switch_l = 0
        steps_since_switch_r = 0
        MIN_STATE_DURATION_STEPS = 25  # Increased to 25 steps (0.5s) to filter 6-10 step glitches seen in logs.

        # Home-gated switch state
        switch_candidate_l = None
        switch_candidate_r = None
        switch_candidate_count_l = 0
        switch_candidate_count_r = 0
        waiting_home_target_l = None
        waiting_home_target_r = None
        home_pause_count_l = 0
        home_pause_count_r = 0
        settle_after_switch_l = 0
        settle_after_switch_r = 0

        # Subtask-level rehome state machine (ported from hierarchical_policy.py)
        subtask_rehome_stage = 'none'  # none | wait_near_home | smooth_to_home
        subtask_rehome_trigger_pending = False
        subtask_rehome_settle_left = 0
        subtask_rehome_detect_threshold = 0.1
        subtask_rehome_interp_delta = 0.01
        subtask_rehome_steps = 50
        subtask_post_lock_steps = 0

        # Conditional Temporal Ensembling: Transition Window Tracking
        transition_window_active_l = 0
        transition_window_active_r = 0
        
        # Display Logic State
        last_removal_step = -100
        current_target_indices_i = []
        current_target_indices_c = []
        
        left_waiting_indep_target = False
        left_resume_reinfer_pending = False
        right_waiting_indep_target = False
        right_resume_reinfer_pending = False
        
        while True:
            # Check input (for quit)
            quit_key = None
            if old_settings:
                quit_key = get_hitl_key()
            if quit_key == 'q':
                break

            # Current joint state for this control step
            qpos_numpy = np.array(ts.observation['qpos'])

            # Start subtask rehome flow on next step after object removal trigger
            if subtask_rehome_trigger_pending and subtask_rehome_stage == 'none':
                subtask_rehome_stage = 'wait_near_home'
                subtask_rehome_trigger_pending = False

            if subtask_rehome_stage == 'wait_near_home':
                left_near = is_arm_at_home(qpos_numpy[:7], home_pose[:7], threshold=subtask_rehome_detect_threshold)
                right_near = is_arm_at_home(qpos_numpy[7:14], home_pose[7:14], threshold=subtask_rehome_detect_threshold)
                if left_near and right_near:
                    subtask_rehome_stage = 'smooth_to_home'
                    subtask_rehome_settle_left = subtask_rehome_steps

            # Arm-wise high-level update ticks
            hl_update_tick_l = (t == 0)
            hl_update_tick_r = (t == 0)
            left_home = None
            right_home = None
            left_open = None
            right_open = None
            hl_update_reason = "init"
            if args.hl_update_at_home_only and not (hl_update_tick_l or hl_update_tick_r):
                home_ok, diff_l, diff_r, left_open, right_open = is_at_home(
                    qpos_numpy,
                    threshold=args.hl_home_threshold,
                    gripper_threshold=args.hl_home_gripper_threshold,
                    return_details=True,
                )
                left_home = (diff_l < args.hl_home_threshold)
                right_home = (diff_r < args.hl_home_threshold)
                left_ready = left_home and left_open
                right_ready = right_home and right_open
                interval_ready = (t % args.hl_update_interval == 0)
                hl_update_tick_l = left_ready and interval_ready
                hl_update_tick_r = right_ready and interval_ready
                hl_update_reason = "home+interval"

                # Periodic diagnostics for non-fired updates
                if args.debug_hl_logs and (not hl_update_tick_l and not hl_update_tick_r) and (t % args.hl_update_log_interval == 0):
                    print(
                        f"[Step {t}] HL not fired | left_home={left_home} right_home={right_home} "
                        f"left_open={left_open} right_open={right_open} "
                        f"interval_ready={interval_ready} "
                        f"since_last_l={t - max(last_hl_update_step_l, 0)} "
                        f"since_last_r={t - max(last_hl_update_step_r, 0)}"
                    )

            if args.debug_hl_logs and args.hl_update_at_home_only and (hl_update_tick_l or hl_update_tick_r):
                if left_home is None or right_home is None:
                    home_ok, diff_l, diff_r, left_open, right_open = is_at_home(
                        qpos_numpy,
                        threshold=args.hl_home_threshold,
                        gripper_threshold=args.hl_home_gripper_threshold,
                        return_details=True,
                    )
                    left_home = (diff_l < args.hl_home_threshold)
                    right_home = (diff_r < args.hl_home_threshold)
                print(
                    f"[Step {t}] HL update fired ({hl_update_reason}) | "
                    f"L_tick={hl_update_tick_l} R_tick={hl_update_tick_r} "
                    f"left_home={left_home} right_home={right_home} "
                    f"left_open={left_open} right_open={right_open} interval={args.hl_update_interval}"
                )
            
            # --- SHADOW SYNC & HIDING ---
            # 1. Sync
            sync_envs(env.physics, env_shadow_c.physics)
            sync_envs(env.physics, env_shadow_i_left.physics)
            sync_envs(env.physics, env_shadow_i_right.physics)
            
            # 2. Heuristic Targets
            color_seq = getattr(env.task, 'color_sequence', MANYCUBES_COLORS[0])
            # Cooperative target update timing
            # - home-only mode: same tick as high-level updates
            # - default mode: original cooldown behavior
            if args.hl_update_at_home_only:
                should_update_targets = (hl_update_tick_l or hl_update_tick_r) 
            else:
                should_update_targets = (t - last_removal_step >= 50)

            # Independent R targets are updated every step.
            spatial_map = get_spatial_object_map(env.physics)
            latest_target_indices_i, latest_target_indices_c = get_heuristic_targets(env.physics, spatial_map, color_seq)
            current_target_indices_i = latest_target_indices_i
            if should_update_targets:
                current_target_indices_c = latest_target_indices_c

            # --- FIX: Keep grasped objects visible ---
            # Ensure held objects don't disappear when moved out of heuristic zones
            grasped_dict = get_grasped_cubes(env.physics)
            held_indices_left = set(grasped_dict['left'])
            held_indices_right = set(grasped_dict['right'])
            held_indices = held_indices_left.union(held_indices_right)
            
            def safe_get_color_local(idx):
                try: return color_seq[idx]
                except: return None

            # Independent display candidates: all non-removed cubes.
            def safe_cube_x_local(idx):
                try:
                    bid = env.physics.model.name2id(f'cube_{idx}', 'body')
                    return float(env.physics.data.xpos[bid][0])
                except Exception:
                    return None

            def is_removed_cube_local(idx):
                try:
                    geom_id = env.physics.model.name2id(f'cube_{idx}', 'geom')
                    alpha = float(env.physics.model.geom_rgba[geom_id, 3])
                except Exception:
                    alpha = 1.0
                try:
                    body_id = env.physics.model.name2id(f'cube_{idx}', 'body')
                    z_pos = float(env.physics.data.xpos[body_id][2])
                except Exception:
                    z_pos = 0.0
                return (alpha <= 0.01) or (z_pos < -1.0)

            final_target_indices_i = [
                idx for idx in range(10)
                if not is_removed_cube_local(idx)
            ]

            def pick_single_indep_target(arm, candidates, exclude=None):
                if exclude is None:
                    exclude = set()

                # Split by x=0 and choose the rightmost (max-X) object on that side.
                side_candidates = []
                for idx in candidates:
                    if idx in exclude:
                        continue
                    if safe_get_color_local(idx) != 'r':
                        continue
                    x_val = safe_cube_x_local(idx)
                    if x_val is None:
                        continue

                    if arm == 'left':
                        if x_val < -0.05:
                            side_candidates.append((x_val, idx))
                    else:
                        if x_val >= -0.05:
                            side_candidates.append((x_val, idx))

                if side_candidates:
                    best_idx = max(side_candidates, key=lambda t_: t_[0])[1]
                    return [best_idx]

                # If no candidate exists on this side, show none.
                return []

            final_target_indices_i_left = pick_single_indep_target('left', final_target_indices_i)
            final_target_indices_i_right = pick_single_indep_target(
                'right',
                final_target_indices_i,
                exclude=set(final_target_indices_i_left),
            )

            def right_indep_target_ready():
                if len(final_target_indices_i_right) == 0:
                    return False, []
                x_vals = []
                for idx in final_target_indices_i_right:
                    x_val = safe_cube_x_local(idx)
                    if x_val is None:
                        return False, []
                    x_vals.append(x_val)
                return all(x >= 0.0 for x in x_vals), x_vals

            def left_indep_target_ready():
                if len(final_target_indices_i_left) == 0:
                    return False, []
                x_vals = []
                for idx in final_target_indices_i_left:
                    x_val = safe_cube_x_local(idx)
                    if x_val is None:
                        return False, []
                    x_vals.append(x_val)
                return all(x >= -0.35 for x in x_vals), x_vals

            # Keep grasped RED cubes visible per arm in independent shadows.
            for idx in held_indices_left:
                if is_removed_cube_local(idx):
                    continue
                if safe_get_color_local(idx) == 'r' and idx not in final_target_indices_i_left:
                    final_target_indices_i_left.append(idx)

            for idx in held_indices_right:
                if is_removed_cube_local(idx):
                    continue
                if safe_get_color_local(idx) == 'r' and idx not in final_target_indices_i_right:
                    final_target_indices_i_right.append(idx)

            # Drop stale coop pairs before display selection.
            if coop_display_lock_pair is not None:
                if is_removed_cube_local(coop_display_lock_pair[0]) or is_removed_cube_local(coop_display_lock_pair[1]):
                    coop_display_lock_pair = None
            if active_coop_pair is not None:
                if is_removed_cube_local(active_coop_pair[0]) or is_removed_cube_local(active_coop_pair[1]):
                    active_coop_pair = None
                
            # Cooperative view must be atomic: exactly one G + one B (or empty).
            # If a pair is latched for delayed removal, keep showing that pair until removed.
            if coop_display_lock_pair is not None:
                final_target_indices_c = [coop_display_lock_pair[0], coop_display_lock_pair[1]]
            else:
                # Keep showing the currently active coop pair until it is removed.
                if active_coop_pair is not None:
                    final_target_indices_c = [active_coop_pair[0], active_coop_pair[1]]
                else:
                    # Build candidate pool from heuristic + currently held non-red cubes,
                    # then pick one G and one B deterministically.
                    coop_pool = set(current_target_indices_c)
                    for h_idx in held_indices:
                        if safe_get_color_local(h_idx) in ['g', 'b']:
                            coop_pool.add(h_idx)

                    g_candidates = []
                    b_candidates = []
                    for c_idx in coop_pool:
                        color = safe_get_color_local(c_idx)
                        if color not in ['g', 'b']:
                            continue
                        try:
                            bid = env.physics.model.name2id(f'cube_{c_idx}', 'body')
                            x_pos = env.physics.data.xpos[bid][0]
                        except Exception:
                            x_pos = -1e9
                        if color == 'g':
                            g_candidates.append((x_pos, c_idx))
                        else:
                            b_candidates.append((x_pos, c_idx))

                    if g_candidates and b_candidates:
                        g_best = max(g_candidates, key=lambda x: x[0])[1]
                        b_best = max(b_candidates, key=lambda x: x[0])[1]
                        active_coop_pair = (g_best, b_best)
                        final_target_indices_c = [g_best, b_best]
                    else:
                        final_target_indices_c = []
            
            if t % 50 == 0:
                 print(
                     f"  [Heuristic Targets] IndepRaw={final_target_indices_i}, "
                     f"IndepL={final_target_indices_i_left}, IndepR={final_target_indices_i_right}, "
                     f"Coop={final_target_indices_c}"
                 )

            # 3. Hide
            hide_objects(env_shadow_i_left.physics, final_target_indices_i_left, "IndepLeft")
            hide_objects(env_shadow_i_right.physics, final_target_indices_i_right, "IndepRight")
            hide_objects(env_shadow_c.physics, final_target_indices_c, "Coop")

            # Render update (optimized: every 5 frames)
            if onscreen_render and t % 5 == 0:
                # Exit if window is closed
                if not plt.get_fignums():
                    print("Window closed. Exiting...")
                    break
                    
                image = env._physics.render(height=240, width=320, camera_id='top')
                plt_img_main.set_data(image)
                
                # Shadow Views
                img_c = env_shadow_c.physics.render(height=240, width=320, camera_id='top')
                img_i_l = env_shadow_i_left.physics.render(height=240, width=320, camera_id='top')
                img_i_r = env_shadow_i_right.physics.render(height=240, width=320, camera_id='top')
                plt_img_c.set_data(img_c)
                plt_img_i_l.set_data(img_i_l)
                plt_img_i_r.set_data(img_i_r)

                # Non-blocking update (prevents focus stealing)
                fig.canvas.draw()
                fig.canvas.flush_events()
                time.sleep(DT)

            # --- State Classifier Inference / Oracle Replay ---
            if current_hl_oracle_schedule is not None:
                future_steps = max(0, int(args.hl_oracle_future_steps))
                if t < future_steps:
                    oracle_query_t = future_steps
                else:
                    oracle_query_t = t + future_steps
                (oracle_l, oracle_r), oracle_pair = _oracle_states_at_t(current_hl_oracle_schedule, oracle_query_t)
                hl_mode_l = oracle_l
                hl_mode_r = oracle_r
                last_hl_update_step_l = t
                last_hl_update_step_r = t
                if t % 50 == 0:
                    print(f"[Step {t}] HL oracle pair={oracle_pair} (query_t={oracle_query_t})")
            else:
                if args.hl_update_at_home_only:
                    should_update_hl = (hl_update_tick_l or hl_update_tick_r)
                else:
                    should_update_hl = True

                if should_update_hl:
                    raw_img_np = ts.observation['images']['top'] # (H,W,C)
                    raw_pil = Image.fromarray(raw_img_np.astype('uint8'))
                    cls_input = cls_transform(raw_pil).unsqueeze(0).cuda()

                    with torch.no_grad():
                        out_l, out_r = state_classifier(cls_input)
                        _, pred_l = torch.max(out_l, 1)
                        _, pred_r = torch.max(out_r, 1)
                        pred_l_item = pred_l.item()
                        pred_r_item = pred_r.item()

                    if (not args.hl_update_at_home_only) or hl_update_tick_l:
                        hl_mode_l = pred_l_item
                        last_hl_update_step_l = t
                    if (not args.hl_update_at_home_only) or hl_update_tick_r:
                        hl_mode_r = pred_r_item
                        last_hl_update_step_r = t

            s_l = hl_mode_l
            s_r = hl_mode_r
            
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

            # --- Pair interpretation rules (user-defined) ---
            # 1) H/C or C/H is not allowed -> interpret as C/C.
            # 2) I/C or C/I is allowed only when transitioning from committed C/C.
            raw_pair = (plan_l_state, plan_r_state)
            committed_pair_before_rules = (committed_plan_l_state, committed_plan_r_state)

            if raw_pair in [('HOLD', 'COOP'), ('COOP', 'HOLD')]:
                plan_l_state, plan_r_state = 'COOP', 'COOP'
                if t % 20 == 0:
                    print(f"[Step {t}] Pair rule: {raw_pair[0]}/{raw_pair[1]} -> COOP/COOP")

            mixed_pair = (plan_l_state, plan_r_state)
            if mixed_pair in [('INDEP', 'COOP'), ('COOP', 'INDEP')] and committed_pair_before_rules != ('COOP', 'COOP'):
                plan_l_state, plan_r_state = committed_pair_before_rules
                if t % 20 == 0:
                    print(
                        f"[Step {t}] Pair rule: {mixed_pair[0]}/{mixed_pair[1]} blocked "
                        f"(committed={committed_pair_before_rules[0]}/{committed_pair_before_rules[1]}) "
                        f"-> {plan_l_state}/{plan_r_state}"
                    )

            # --- Home-conditioned state-pair constraints ---
            # User rule:
            # - Exactly one arm at home: allowed pairs are IC / CI / II
            # - Both arms at home: allowed pairs are II / CC / HI / IH
            # (I=INDEP, C=COOP, H=HOLD)
            left_arm_home_for_pair = is_arm_at_home(
                qpos_numpy[:7],
                home_pose[:7],
                threshold=args.switch_home_threshold,
            )
            right_arm_home_for_pair = is_arm_at_home(
                qpos_numpy[7:14],
                home_pose[7:14],
                threshold=args.switch_home_threshold,
            )

            current_pair = (plan_l_state, plan_r_state)
            allowed_pairs = None

            if left_arm_home_for_pair ^ right_arm_home_for_pair:
                allowed_pairs = {
                    ('INDEP', 'COOP'),
                    ('COOP', 'INDEP'),
                    ('INDEP', 'INDEP'),
                }
            elif left_arm_home_for_pair and right_arm_home_for_pair:
                allowed_pairs = {
                    ('INDEP', 'INDEP'),
                    ('COOP', 'COOP'),
                    ('HOLD', 'INDEP'),
                    ('INDEP', 'HOLD'),
                }

            # In HL-oracle replay mode, trust oracle pair transitions and do not override
            # them with home-conditioned pair constraints.
            if current_hl_oracle_schedule is None and allowed_pairs is not None and current_pair not in allowed_pairs:
                # Prefer maintaining current committed execution state when constraining.
                # Score candidates by how many sides match committed plans (higher is better),
                # then how many sides match current proposal as tie-breaker.
                committed_pair = (committed_plan_l_state, committed_plan_r_state)
                ranked = sorted(
                    list(allowed_pairs),
                    key=lambda p: (
                        int(p[0] == committed_pair[0]) + int(p[1] == committed_pair[1]),
                        int(p[0] == current_pair[0]) + int(p[1] == current_pair[1]),
                    ),
                    reverse=True,
                )
                plan_l_state, plan_r_state = ranked[0]
                if t % 20 == 0:
                    print(
                        f"[Step {t}] Pair constrained by home state: "
                        f"{current_pair[0]}/{current_pair[1]} -> {plan_l_state}/{plan_r_state} "
                        f"(left_home={left_arm_home_for_pair}, right_home={right_arm_home_for_pair})"
                    )

            # Episode-start alignment:
            # At t=0, use fresh classifier-based plan immediately instead of inheriting
            # default committed INDEP from previous initialization.
            if t == 0:
                committed_plan_l_state = plan_l_state
                committed_plan_r_state = plan_r_state
                steps_since_switch_l = 0
                steps_since_switch_r = 0
                switch_candidate_l = None
                switch_candidate_r = None
                switch_candidate_count_l = 0
                switch_candidate_count_r = 0
                waiting_home_target_l = None
                waiting_home_target_r = None
                home_pause_count_l = 0
                home_pause_count_r = 0
            
            # --- Debounce / Latching Logic ---
            steps_since_switch_l += 1
            proposed_l_state = plan_l_state # Store raw intent
            
            if plan_l_state != committed_plan_l_state:
                if steps_since_switch_l > MIN_STATE_DURATION_STEPS:
                     committed_plan_l_state = plan_l_state
                     steps_since_switch_l = 0
                else:
                     # Suppress switch - stay committed
                     plan_l_state = committed_plan_l_state

            steps_since_switch_r += 1
            proposed_r_state = plan_r_state # Store raw intent
            
            if plan_r_state != committed_plan_r_state:
                if steps_since_switch_r > MIN_STATE_DURATION_STEPS:
                     committed_plan_r_state = plan_r_state
                     steps_since_switch_r = 0
                else:
                     # Suppress switch - stay committed
                     plan_r_state = committed_plan_r_state

            # --- Home-Gated Switching Logic ---
            # Don't switch immediately on high-level prediction.
            # Require stable prediction window, then wait until arm reaches home,
            # pause briefly, clear buffers, and only then activate new policy.
            force_hold_l = False
            force_hold_r = False

            # LEFT arm: track stable switch request
            if proposed_l_state in ['INDEP', 'COOP'] and proposed_l_state != committed_plan_l_state:
                if switch_candidate_l == proposed_l_state:
                    switch_candidate_count_l += 1
                else:
                    switch_candidate_l = proposed_l_state
                    switch_candidate_count_l = 1
            else:
                switch_candidate_l = None
                switch_candidate_count_l = 0

            if waiting_home_target_l is None and switch_candidate_l is not None and switch_candidate_count_l >= args.switch_guard_steps:
                waiting_home_target_l = switch_candidate_l
                home_pause_count_l = 0
                print(f"[Step {t}] Left switch requested: {committed_plan_l_state} -> {waiting_home_target_l} (home-gated)")

            # RIGHT arm: track stable switch request
            if proposed_r_state in ['INDEP', 'COOP'] and proposed_r_state != committed_plan_r_state:
                if switch_candidate_r == proposed_r_state:
                    switch_candidate_count_r += 1
                else:
                    switch_candidate_r = proposed_r_state
                    switch_candidate_count_r = 1
            else:
                switch_candidate_r = None
                switch_candidate_count_r = 0

            if waiting_home_target_r is None and switch_candidate_r is not None and switch_candidate_count_r >= args.switch_guard_steps:
                waiting_home_target_r = switch_candidate_r
                home_pause_count_r = 0
                print(f"[Step {t}] Right switch requested: {committed_plan_r_state} -> {waiting_home_target_r} (home-gated)")

            # LEFT arm: hold current policy until home reached, then pause and switch
            if waiting_home_target_l is not None:
                if is_arm_at_home(qpos_numpy[:7], home_pose[:7], threshold=args.switch_home_threshold):
                    if home_pause_count_l < args.switch_home_pause_steps:
                        force_hold_l = True
                        plan_l_state = 'HOLD'
                        proposed_l_state = 'HOLD'
                        home_pause_count_l += 1
                    else:
                        committed_plan_l_state = waiting_home_target_l
                        plan_l_state = committed_plan_l_state
                        proposed_l_state = committed_plan_l_state
                        steps_since_switch_l = 0

                        waiting_home_target_l = None
                        switch_candidate_l = None
                        switch_candidate_count_l = 0
                        home_pause_count_l = 0

                        all_time_actions_left.fill_(float_nan)
                        all_time_actions_dual[:, :, :7].fill_(float_nan)
                        current_action_chunk_left = None
                        current_action_chunk_dual = None
                        transition_window_active_l = 0
                        step_in_chunk_left = 0
                        step_in_chunk_dual = 0
                        settle_after_switch_l = max(0, args.switch_home_settle_steps)
                        print(f"[Step {t}] Left switch activated at home -> {plan_l_state}")
                else:
                    plan_l_state = committed_plan_l_state
                    proposed_l_state = committed_plan_l_state

            # RIGHT arm: hold current policy until home reached, then pause and switch
            if waiting_home_target_r is not None:
                if is_arm_at_home(qpos_numpy[7:14], home_pose[7:14], threshold=args.switch_home_threshold):
                    if home_pause_count_r < args.switch_home_pause_steps:
                        force_hold_r = True
                        plan_r_state = 'HOLD'
                        proposed_r_state = 'HOLD'
                        home_pause_count_r += 1
                    else:
                        committed_plan_r_state = waiting_home_target_r
                        plan_r_state = committed_plan_r_state
                        proposed_r_state = committed_plan_r_state
                        steps_since_switch_r = 0

                        waiting_home_target_r = None
                        switch_candidate_r = None
                        switch_candidate_count_r = 0
                        home_pause_count_r = 0

                        all_time_actions_right.fill_(float_nan)
                        all_time_actions_dual[:, :, 7:].fill_(float_nan)
                        current_action_chunk_right = None
                        current_action_chunk_dual = None
                        transition_window_active_r = 0
                        step_in_chunk_right = 0
                        step_in_chunk_dual = 0
                        settle_after_switch_r = max(0, args.switch_home_settle_steps)
                        print(f"[Step {t}] Right switch activated at home -> {plan_r_state}")
                else:
                    plan_r_state = committed_plan_r_state
                    proposed_r_state = committed_plan_r_state
            
            # --- Pre-Switch Transition Blending Logic ---
            # Detect if we are in the "final approach" of a switch
            TRANSITION_WINDOW = 20
            
            # LEFT
            is_transitioning_l = False
            alpha_l = 0.0
            if plan_l_state != proposed_l_state: # Switch is currently suppressed
                steps_remaining = MIN_STATE_DURATION_STEPS - steps_since_switch_l
                if steps_remaining <= TRANSITION_WINDOW:
                    is_transitioning_l = True
                    # alpha goes 0 -> 1 as steps_remaining goes 20 -> 0
                    alpha_l = (TRANSITION_WINDOW - steps_remaining) / TRANSITION_WINDOW
                    # print(f"DEBUG: Smooth Left {plan_l_state}->{proposed_l_state} alpha={alpha_l:.2f}")

            # RIGHT
            is_transitioning_r = False
            alpha_r = 0.0
            if plan_r_state != proposed_r_state:
                steps_remaining = MIN_STATE_DURATION_STEPS - steps_since_switch_r
                if steps_remaining <= TRANSITION_WINDOW:
                    is_transitioning_r = True
                    alpha_r = (TRANSITION_WINDOW - steps_remaining) / TRANSITION_WINDOW
            
            # ---------------------------------
            
            # --- Smart HOLD Logic: high-level HOLD should smoothly return home ---
            # Do not keep executing previous active policy when HOLD is predicted.
            
            # Left Arm
            if force_hold_l:
                plan_l_state = 'HOLD'
            elif plan_l_state == 'HOLD':
                # Keep HOLD; action stage performs smooth interpolation to home.
                consecutive_at_home_l = 0
            else:
                consecutive_at_home_l = 0
            
            # Right Arm
            if force_hold_r:
                plan_r_state = 'HOLD'
            elif plan_r_state == 'HOLD':
                # Keep HOLD; action stage performs smooth interpolation to home.
                consecutive_at_home_r = 0
            else:
                consecutive_at_home_r = 0

            # During rehome wait stage, keep current committed policies (same as hierarchical behavior)
            if subtask_rehome_stage == 'wait_near_home':
                plan_l_state = committed_plan_l_state
                plan_r_state = committed_plan_r_state
                proposed_l_state = committed_plan_l_state
                proposed_r_state = committed_plan_r_state

            # After smooth rehome completion, briefly freeze to committed plans.
            if subtask_post_lock_steps > 0:
                plan_l_state = committed_plan_l_state
                plan_r_state = committed_plan_r_state
                proposed_l_state = committed_plan_l_state
                proposed_r_state = committed_plan_r_state

            left_home_gate_threshold = max(args.switch_home_threshold, args.hl_home_threshold)
            left_home_now = np.max(np.abs(qpos_numpy[:6] - home_pose[:6])) < left_home_gate_threshold
            left_indep_intent = (
                (waiting_home_target_l == 'INDEP')
                or (proposed_l_state == 'INDEP')
            )

            if left_waiting_indep_target:
                ready_l, target_x_vals_l = left_indep_target_ready()
                if ready_l:
                    print(f"[Step {t}] Left INDEP resume: left-shadow target arrived (x={target_x_vals_l})")
                    left_waiting_indep_target = False
                    force_hold_l = False
                    plan_l_state = 'INDEP'
                    proposed_l_state = 'INDEP'
                    committed_plan_l_state = 'INDEP'
                    steps_since_switch_l = 0
                    all_time_actions_left.fill_(float_nan)
                    all_time_actions_dual[:, :, :7].fill_(float_nan)
                    current_action_chunk_left = None
                    step_in_chunk_left = 0
                    transition_window_active_l = 0
                    left_resume_reinfer_pending = True
                elif left_indep_intent:
                    force_hold_l = True
                    plan_l_state = 'HOLD'
                    proposed_l_state = 'HOLD'
                else:
                    left_waiting_indep_target = False
            elif waiting_home_target_l is None and left_indep_intent and left_home_now:
                ready_l, target_x_vals_l = left_indep_target_ready()
                if not ready_l:
                    force_hold_l = True
                    plan_l_state = 'HOLD'
                    proposed_l_state = 'HOLD'
                    if len(target_x_vals_l) == 0:
                        print(f"[Step {t}] Left INDEP wait at home: no left-shadow target -> HOLD")
                    else:
                        print(f"[Step {t}] Left INDEP wait at home: shadow target x={target_x_vals_l} -> HOLD")
                    left_waiting_indep_target = True

            right_home_gate_threshold = max(args.switch_home_threshold, args.hl_home_threshold)
            right_home_now = np.max(np.abs(qpos_numpy[7:13] - home_pose[7:13])) < right_home_gate_threshold
            right_indep_intent = (
                (waiting_home_target_r == 'INDEP')
                or (proposed_r_state == 'INDEP')
            )

            if right_waiting_indep_target:
                ready_r, target_x_vals_r = right_indep_target_ready()
                if ready_r:
                    print(f"[Step {t}] Right INDEP resume: right-shadow target arrived (x={target_x_vals_r})")
                    right_waiting_indep_target = False
                    force_hold_r = False
                    plan_r_state = 'INDEP'
                    proposed_r_state = 'INDEP'
                    committed_plan_r_state = 'INDEP'
                    steps_since_switch_r = 0
                    all_time_actions_right.fill_(float_nan)
                    all_time_actions_dual[:, :, 7:].fill_(float_nan)
                    current_action_chunk_right = None
                    step_in_chunk_right = 0
                    transition_window_active_r = 0
                    right_resume_reinfer_pending = True
                elif right_indep_intent:
                    force_hold_r = True
                    plan_r_state = 'HOLD'
                    proposed_r_state = 'HOLD'
                else:
                    right_waiting_indep_target = False
            elif waiting_home_target_r is None and right_indep_intent and right_home_now:
                ready_r, target_x_vals_r = right_indep_target_ready()
                if not ready_r:
                    force_hold_r = True
                    plan_r_state = 'HOLD'
                    proposed_r_state = 'HOLD'
                    if len(target_x_vals_r) == 0:
                        print(f"[Step {t}] Right INDEP wait at home: no right-shadow target -> HOLD")
                    else:
                        print(f"[Step {t}] Right INDEP wait at home: shadow target x={target_x_vals_r} -> HOLD")
                    right_waiting_indep_target = True

            # Logging
            if args.debug_hl_logs and t % 50 == 0:
                print(f"[Step {t}] Classifier: L={STATE_NAMES[s_l]} R={STATE_NAMES[s_r]} -> Plan: L={plan_l_state} R={plan_r_state} (Agg: {temporal_agg})")
                if onscreen_render:
                    plt.title(f"Plan: L={plan_l_state} R={plan_r_state} (State: {STATE_NAMES[s_l]}/{STATE_NAMES[s_r]})")

            # --- Buffer Inheritance / State Tracking Logic ---
            if args.inherit_temporal_buffer and temporal_agg:
                # LEFT transition checking
                # Trigger inheritance if we ENTER a moving state (INDEP/COOP) from a DIFFERENT moving state (even with HOLD between)
                if plan_l_state in ['INDEP', 'COOP']:
                    if last_active_l_state is not None and last_active_l_state != plan_l_state:
                        # Transition detected!
                        if plan_l_state == 'COOP':
                            # Inherit from INDEP
                            # print(f"[Step {t}] Left Inheritance: {last_active_l_state} -> COOP")
                            input_actions = all_time_actions_left
                            mask_val = ~torch.isnan(input_actions)
                            input_safe = torch.nan_to_num(input_actions, nan=0.0)
                            denorm_l = input_safe * stats_left_torch['action_std'] + stats_left_torch['action_mean']
                            renorm_dual_l = (denorm_l - stats_dual_torch['action_mean'][:7]) / stats_dual_torch['action_std'][:7]
                            val_inherit = renorm_dual_l.clone()
                            val_inherit[~mask_val] = float('nan') 
                            all_time_actions_dual[:, :, :7] = val_inherit
                        else:
                            # Inherit from COOP
                            # print(f"[Step {t}] Left Inheritance: {last_active_l_state} -> INDEP")
                            input_actions = all_time_actions_dual[:, :, :7]
                            mask_val = ~torch.isnan(input_actions)
                            input_safe = torch.nan_to_num(input_actions, nan=0.0)
                            denorm_dual_l = input_safe * stats_dual_torch['action_std'][:7] + stats_dual_torch['action_mean'][:7]
                            renorm_l = (denorm_dual_l - stats_left_torch['action_mean']) / stats_left_torch['action_std']
                            val_inherit = renorm_l.clone()
                            val_inherit[~mask_val] = float('nan')
                            all_time_actions_left.copy_(val_inherit)
                        # Activate transition window for conditional temporal ensembling
                        if args.temporal_agg_transition_only:
                            transition_window_active_l = 2 * args.temporal_agg_window
                            print(f"[Step {t}] Left transition {last_active_l_state} -> {plan_l_state}, temporal agg window: {transition_window_active_l} steps")
                    # Update last active moving state
                    last_active_l_state = plan_l_state
                
                # RIGHT transition checking
                if plan_r_state in ['INDEP', 'COOP']:
                    if last_active_r_state is not None and last_active_r_state != plan_r_state:
                        if plan_r_state == 'COOP':
                            # print(f"[Step {t}] Right Inheritance: {last_active_r_state} -> COOP")
                            input_actions = all_time_actions_right
                            mask_val = ~torch.isnan(input_actions)
                            input_safe = torch.nan_to_num(input_actions, nan=0.0)
                            denorm_r = input_safe * stats_right_torch['action_std'] + stats_right_torch['action_mean']
                            renorm_dual_r = (denorm_r - stats_dual_torch['action_mean'][7:]) / stats_dual_torch['action_std'][7:]
                            val_inherit = renorm_dual_r.clone()
                            val_inherit[~mask_val] = float('nan')
                            all_time_actions_dual[:, :, 7:] = val_inherit
                        else:
                            # print(f"[Step {t}] Right Inheritance: {last_active_r_state} -> INDEP")
                            input_actions = all_time_actions_dual[:, :, 7:]
                            mask_val = ~torch.isnan(input_actions)
                            input_safe = torch.nan_to_num(input_actions, nan=0.0)
                            denorm_dual_r = input_safe * stats_dual_torch['action_std'][7:] + stats_dual_torch['action_mean'][7:]
                            renorm_r = (denorm_dual_r - stats_right_torch['action_mean']) / stats_right_torch['action_std']
                            val_inherit = renorm_r.clone()
                            val_inherit[~mask_val] = float('nan')
                            all_time_actions_right.copy_(val_inherit)
                        
                        # Activate transition window for conditional temporal ensembling
                        if args.temporal_agg_transition_only:
                            transition_window_active_r = 2 * args.temporal_agg_window
                            print(f"[Step {t}] Right transition {last_active_r_state} -> {plan_r_state}, temporal agg window: {transition_window_active_r} steps")
                    
                    # Update last active moving state
                    last_active_r_state = plan_r_state


            # -----------------------
            # 4.3 HUMAN-IN-THE-LOOP OVERRIDE
            # -----------------------
            key = get_hitl_key()
            if key:
                if key == 'q':   hitl_mode_l, hitl_mode_r = 'HOLD', 'INDEP'
                elif key == 'w': hitl_mode_l, hitl_mode_r = 'INDEP', 'INDEP'
                elif key == 'e': hitl_mode_l, hitl_mode_r = 'INDEP', 'HOLD'
                elif key == 'a': hitl_mode_l, hitl_mode_r = 'INDEP', 'COOP'
                elif key == 's': hitl_mode_l, hitl_mode_r = 'COOP', 'COOP'
                elif key == 'd': hitl_mode_l, hitl_mode_r = 'COOP', 'INDEP'
                elif key == ' ': hitl_mode_l, hitl_mode_r = 'AUTO', 'AUTO'
                elif key == 't': 
                    temporal_agg = not temporal_agg
                    print(f" [System] Temporal Aggregation: {temporal_agg}        ")
                
            # Apply Override
            is_override = False
            if hitl_mode_l != 'AUTO':
                 plan_l_state = hitl_mode_l
                 # Force commit to prevent debounce from reverting
                 committed_plan_l_state = plan_l_state 
                 is_override = True
            if hitl_mode_r != 'AUTO':
                 plan_r_state = hitl_mode_r
                 committed_plan_r_state = plan_r_state
                 is_override = True
                 
            if is_override:
                 print(f" [HITL override] {hitl_mode_l}/{hitl_mode_r}", end='\r')


            # Determine Final Effective State for Saving
            effective_l = plan_l_state
            effective_r = plan_r_state
            video_mode_l_char = mode_to_char(effective_l)
            video_mode_r_char = mode_to_char(effective_r)

            if args.save_video:
                main_rgb = env._physics.render(height=240, width=320, camera_id='top')
                coop_rgb = env_shadow_c.physics.render(height=240, width=320, camera_id='top')
                indep_l_rgb = env_shadow_i_left.physics.render(height=240, width=320, camera_id='top')
                indep_r_rgb = env_shadow_i_right.physics.render(height=240, width=320, camera_id='top')
                video_frame = compose_video_frame(
                    main_rgb,
                    coop_rgb,
                    indep_l_rgb,
                    indep_r_rgb,
                    video_mode_l_char,
                    video_mode_r_char,
                )
                video_frames.append(video_frame)
            
            # --- Record HITL Data (if enabled) ---
            if args.save_hitl_data_path:
                curr_img_top = ts.observation['images']['top'] # (H, W, C) uint8
                hitl_images.append(curr_img_top)
                hitl_labels_l.append(state_to_int(effective_l))
                hitl_labels_r.append(state_to_int(effective_r))

            obs = ts.observation
            qpos_numpy = np.array(obs['qpos'])
            
            # --- Dynamic Temporal Aggregation Control ---
            use_temporal_agg_l = temporal_agg
            use_temporal_agg_r = temporal_agg
            force_query_dual = False
            force_query_l = left_resume_reinfer_pending
            force_query_r = right_resume_reinfer_pending
            if left_resume_reinfer_pending:
                left_resume_reinfer_pending = False
            if right_resume_reinfer_pending:
                right_resume_reinfer_pending = False
            
            if args.temporal_agg_transition_only:
                use_temporal_agg_l = (transition_window_active_l > 0)
                use_temporal_agg_r = (transition_window_active_r > 0)
                
                # Decrement counters
                if transition_window_active_l > 0:
                    transition_window_active_l -= 1
                    if transition_window_active_l == 0:
                        # TE finished for Left, force local re-query only
                        force_query_l = True
                if transition_window_active_r > 0:
                    transition_window_active_r -= 1
                    if transition_window_active_r == 0:
                        # TE finished for Right, force local re-query only
                        force_query_r = True

            # Home-only HL mode: at home update ticks, discard remaining chunks and
            # start fresh inference so previous task leftovers do not bleed into next task.
            if args.hl_update_at_home_only:
                refresh_left_indep = hl_update_tick_l and (plan_l_state == 'INDEP')
                refresh_right_indep = hl_update_tick_r and (plan_r_state == 'INDEP')
                refresh_dual_coop = hl_update_tick_l and hl_update_tick_r and (plan_l_state == 'COOP') and (plan_r_state == 'COOP')

                if refresh_left_indep:
                    current_action_chunk_left = None
                    step_in_chunk_left = 0
                    all_time_actions_left.fill_(float_nan)
                    force_query_l = True
                    if args.debug_hl_logs:
                        print(f"[Step {t}] Home refresh: LEFT INDEP chunk dropped -> force re-infer")

                if refresh_right_indep:
                    current_action_chunk_right = None
                    step_in_chunk_right = 0
                    all_time_actions_right.fill_(float_nan)
                    force_query_r = True
                    if args.debug_hl_logs:
                        print(f"[Step {t}] Home refresh: RIGHT INDEP chunk dropped -> force re-infer")

                if refresh_dual_coop:
                    current_action_chunk_dual = None
                    step_in_chunk_dual = 0
                    all_time_actions_dual.fill_(float_nan)
                    force_query_dual = True
                    if args.debug_hl_logs:
                        print(f"[Step {t}] Home refresh: COOP chunk dropped (both home) -> force re-infer")
            with torch.inference_mode():
                if step_in_chunk_dual >= chunk_size and not (use_temporal_agg_l or use_temporal_agg_r):
                    step_in_chunk_dual = 0
                if step_in_chunk_left >= chunk_size and not use_temporal_agg_l:
                    step_in_chunk_left = 0
                if step_in_chunk_right >= chunk_size and not use_temporal_agg_r:
                    step_in_chunk_right = 0

                # Check for mode switch to force replan
                mode_switch_l = (prev_plan_l_state is not None and prev_plan_l_state != plan_l_state)
                mode_switch_r = (prev_plan_r_state is not None and prev_plan_r_state != plan_r_state)

                if mode_switch_l and not use_temporal_agg_l:
                    if prev_plan_l_state in ['INDEP', 'HOLD'] or plan_l_state in ['INDEP', 'HOLD']:
                        force_query_l = True
                    if prev_plan_l_state == 'COOP' or plan_l_state == 'COOP':
                        force_query_dual = True
                if mode_switch_r and not use_temporal_agg_r:
                    if prev_plan_r_state in ['INDEP', 'HOLD'] or plan_r_state in ['INDEP', 'HOLD']:
                        force_query_r = True
                    if prev_plan_r_state == 'COOP' or plan_r_state == 'COOP':
                        force_query_dual = True
                
                prev_plan_l_state = plan_l_state
                prev_plan_r_state = plan_r_state

                # 1. Query Dual (if needed by ANY arm)
                # Query condition: Start of chunk OR Temporal Aggregation Active
                # ALSO Query if transition demands it (Secondary Policy)
                need_dual_l = (plan_l_state == 'COOP') or (is_transitioning_l and proposed_l_state == 'COOP')
                need_dual_r = (plan_r_state == 'COOP') or (is_transitioning_r and proposed_r_state == 'COOP')
                
                should_query_dual = (step_in_chunk_dual == 0) or (need_dual_l and use_temporal_agg_l) or (need_dual_r and use_temporal_agg_r) or force_query_dual

                if (need_dual_l or need_dual_r) and (current_action_chunk_dual is None):
                    should_query_dual = True

                if (need_dual_l or need_dual_r) and should_query_dual:
                     # Prepare input for Dual Policy
                    qpos_numpy_dual = qpos_numpy.copy()
                      
                     # --- GHOST ARM LOGIC (Corrected) ---
                     # Previously we forced the non-coop arm to HOME_POSE.
                     # This caused jumps because the Dual Policy saw "Arm Teleported to Home" and reacted.
                     # CORRECT LOGIC: Pass the REAL qpos of the non-coop arm. 
                     # The policy should be robust enough, or at least it won't see a teleport.
                     # So we DO NOT mask the input anymore.
                     # -------------------------------------------
                     
                    # Shadow Env Observation
                    obs_c = env_shadow_c.task.get_observation(env_shadow_c.physics)
                    ts_c = type('TS', (object,), {'observation': obs_c})()

                    qpos = pre_process_dual(qpos_numpy_dual)
                    qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                    curr_image = get_image_dual(ts_c, camera_names, mask=False)
                      
                    action_chunk = policy_dual(qpos, curr_image) # [1, chunk_size, 14]
                    
                    # Store in buffer if either arm using COOP needs temporal agg
                    if (plan_l_state == 'COOP' and use_temporal_agg_l) or (plan_r_state == 'COOP' and use_temporal_agg_r) or \
                       (is_transitioning_l and proposed_l_state == 'COOP' and use_temporal_agg_l) or \
                       (is_transitioning_r and proposed_r_state == 'COOP' and use_temporal_agg_r):
                        all_time_actions_dual[[t], t:t+num_queries] = action_chunk
                    
                    # ALWAYS update chunk for standardized access (fixes transition crashes)
                    current_action_chunk_dual = action_chunk.squeeze(0).cpu().numpy()
                    step_in_chunk_dual = 0


                # 2. Query Independent Left (if needed)
                need_indep_l = (plan_l_state == 'INDEP') or (is_transitioning_l and proposed_l_state == 'INDEP')
                should_query_l = (step_in_chunk_left == 0) or (need_indep_l and use_temporal_agg_l) or force_query_l
                if need_indep_l and current_action_chunk_left is None: should_query_l = True
                
                # Pre-fetch shadow obs if needed
                need_indep_r = (plan_r_state == 'INDEP') or (is_transitioning_r and proposed_r_state == 'INDEP')
                should_query_r = (step_in_chunk_right == 0) or (need_indep_r and use_temporal_agg_r) or force_query_r
                if need_indep_r and current_action_chunk_right is None: should_query_r = True

                ts_i_left = None
                ts_i_right = None
                if need_indep_l and should_query_l:
                    obs_i_left = env_shadow_i_left.task.get_observation(env_shadow_i_left.physics)
                    ts_i_left = type('TS', (object,), {'observation': obs_i_left})()
                if need_indep_r and should_query_r:
                    obs_i_right = env_shadow_i_right.task.get_observation(env_shadow_i_right.physics)
                    ts_i_right = type('TS', (object,), {'observation': obs_i_right})()

                if need_indep_l and should_query_l:
                    qpos_left_numpy = qpos_numpy[:7]
                    qpos_left = pre_process_left(qpos_left_numpy)
                    qpos_left = torch.from_numpy(qpos_left).float().cuda().unsqueeze(0)
                    curr_image_left = get_image_independent(ts_i_left, camera_names, 'left', mask=False)
                    
                    action_chunk_l = policy_left(qpos_left, curr_image_left)
                    if use_temporal_agg_l:
                         # Store even if transitioning? Yes.
                        all_time_actions_left[[t], t:t+num_queries] = action_chunk_l
                    
                    # ALWAYS update chunk
                    current_action_chunk_left = action_chunk_l.squeeze(0).cpu().numpy()
                    step_in_chunk_left = 0
                
                # 3. Query Independent Right (if needed)
                if need_indep_r and should_query_r:
                    qpos_right_numpy = qpos_numpy[7:14]
                    qpos_right = pre_process_right(qpos_right_numpy)
                    qpos_right = torch.from_numpy(qpos_right).float().cuda().unsqueeze(0)
                    curr_image_right = get_image_independent(ts_i_right, camera_names, 'right', mask=False)
                     
                    action_chunk_r = policy_right(qpos_right, curr_image_right)
                    if use_temporal_agg_r:
                        all_time_actions_right[[t], t:t+num_queries] = action_chunk_r
                    
                    # ALWAYS update chunk
                    current_action_chunk_right = action_chunk_r.squeeze(0).cpu().numpy()
                    step_in_chunk_right = 0
                
                # 4. Anchor HOLD state in buffers to prevent jumps when restarting
                if temporal_agg:
                    if plan_l_state == 'HOLD' and not is_transitioning_l:
                        # Fill future with CURRENT pose (Stay Here intention)
                        q_l_norm = pre_process_left(qpos_numpy[:7])
                        q_l_norm_dual = (qpos_numpy[:7] - stats_dual['qpos_mean'][:7]) / stats_dual['qpos_std'][:7]
                        all_time_actions_left[t, t:t+num_queries] = torch.from_numpy(q_l_norm).cuda()
                        all_time_actions_dual[t, t:t+num_queries, :7] = torch.from_numpy(q_l_norm_dual).cuda()
                        
                    if plan_r_state == 'HOLD' and not is_transitioning_r:
                        q_r_norm = pre_process_right(qpos_numpy[7:14])
                        q_r_norm_dual = (qpos_numpy[7:14] - stats_dual['qpos_mean'][7:14]) / stats_dual['qpos_std'][7:14]
                        all_time_actions_right[t, t:t+num_queries] = torch.from_numpy(q_r_norm).cuda()
                        all_time_actions_dual[t, t:t+num_queries, 7:] = torch.from_numpy(q_r_norm_dual).cuda()
                             
                
                # --- HELPER: Get Action for a specific state ---
                def get_action_for_state(target_state, arm_side, input_qpos_slice):
                    """
                    arm_side: 'left' or 'right'
                    input_qpos_slice: qpos of that arm (7,)
                    Returns: (7,) numpy action
                    """
                    # Left Arm Logic
                    if arm_side == 'left':
                        if target_state == 'COOP':
                            if use_temporal_agg_l:
                                actions = all_time_actions_dual[:, t]
                                valid = torch.all(~torch.isnan(actions[:, :7]), axis=1)
                                actions = actions[valid]
                                if len(actions) == 0: return input_qpos_slice # Fail safe
                                k = args.temporal_agg_k
                                w_len = len(actions)
                                weights = np.exp(-k * (w_len - 1 - np.arange(w_len)))
                                weights = weights / weights.sum()
                                weights = torch.from_numpy(weights).cuda().unsqueeze(dim=1)
                                raw = (actions * weights).sum(dim=0, keepdim=True).squeeze(0).cpu().numpy()[:7]
                            else:
                                safe_step = step_in_chunk_dual if step_in_chunk_dual < chunk_size else 0
                                raw = current_action_chunk_dual[safe_step][:7]
                            return raw * stats_dual['action_std'][:7] + stats_dual['action_mean'][:7]

                        elif target_state == 'INDEP':
                            if use_temporal_agg_l:
                                actions = all_time_actions_left[:, t]
                                valid = torch.all(~torch.isnan(actions), axis=1)
                                actions = actions[valid]
                                if len(actions) == 0:
                                    return input_qpos_slice
                                k = args.temporal_agg_k
                                w_len = len(actions)
                                weights = np.exp(-k * (w_len - 1 - np.arange(w_len)))
                                weights = weights / weights.sum()
                                weights = torch.from_numpy(weights).cuda().unsqueeze(dim=1)
                                raw = (actions * weights).sum(dim=0, keepdim=True).squeeze(0).cpu().numpy()
                            else:
                                safe_step = step_in_chunk_left if step_in_chunk_left < chunk_size else 0
                                raw = current_action_chunk_left[safe_step]
                            return raw * stats_left['action_std'] + stats_left['action_mean']
                        else:
                            return input_qpos_slice # HOLD

                    # Right Arm Logic
                    else:
                        if target_state == 'COOP':
                            if use_temporal_agg_r:
                                actions = all_time_actions_dual[:, t]
                                valid = torch.all(~torch.isnan(actions[:, 7:]), axis=1)
                                actions = actions[valid]
                                if len(actions) == 0: return input_qpos_slice
                                k = args.temporal_agg_k
                                w_len = len(actions)
                                weights = np.exp(-k * (w_len - 1 - np.arange(w_len)))
                                weights = weights / weights.sum()
                                weights = torch.from_numpy(weights).cuda().unsqueeze(dim=1)
                                raw = (actions * weights).sum(dim=0, keepdim=True).squeeze(0).cpu().numpy()[7:]
                            else:
                                safe_step = step_in_chunk_dual if step_in_chunk_dual < chunk_size else 0
                                raw = current_action_chunk_dual[safe_step][7:]
                            return raw * stats_dual['action_std'][7:] + stats_dual['action_mean'][7:]

                        elif target_state == 'INDEP':
                            if use_temporal_agg_r:
                                actions = all_time_actions_right[:, t]
                                valid = torch.all(~torch.isnan(actions), axis=1)
                                actions = actions[valid]
                                if len(actions) == 0: return input_qpos_slice
                                k = args.temporal_agg_k
                                w_len = len(actions)
                                weights = np.exp(-k * (w_len - 1 - np.arange(w_len)))
                                weights = weights / weights.sum()
                                weights = torch.from_numpy(weights).cuda().unsqueeze(dim=1)
                                raw = (actions * weights).sum(dim=0, keepdim=True).squeeze(0).cpu().numpy()
                            else:
                                safe_step = step_in_chunk_right if step_in_chunk_right < chunk_size else 0
                                raw = current_action_chunk_right[safe_step]
                            return raw * stats_right['action_std'] + stats_right['action_mean']
                        else:
                            return input_qpos_slice # HOLD

                def interp_toward_home(curr_qpos_slice, home_qpos_slice):
                    max_step = float(args.switch_home_interp_max_delta)
                    delta = home_qpos_slice - curr_qpos_slice
                    step = np.clip(delta, -max_step, max_step)
                    target = curr_qpos_slice + step
                    if target.shape[0] >= 7:
                        target[-1] = curr_qpos_slice[-1]
                    return target

                # --- 1. Get Left Action ---
                action_l_committed = get_action_for_state(plan_l_state, 'left', qpos_numpy[:7])
                if is_transitioning_l:
                    action_l_proposed = get_action_for_state(proposed_l_state, 'left', qpos_numpy[:7])
                    # Blend: alpha=0 means 100% committed, alpha=1 means 100% proposed
                    action_l = (1 - alpha_l) * action_l_committed + alpha_l * action_l_proposed
                else:
                    action_l = action_l_committed

                # Smoothly move toward home during HOLD (including HL HOLD) and switch hold/settle
                if ((plan_l_state == 'HOLD' and not is_transitioning_l) or force_hold_l or settle_after_switch_l > 0):
                    action_l = interp_toward_home(qpos_numpy[:7], home_pose[:7])
                    if settle_after_switch_l > 0:
                        settle_after_switch_l -= 1

                # --- 2. Get Right Action ---
                action_r_committed = get_action_for_state(plan_r_state, 'right', qpos_numpy[7:14])
                if is_transitioning_r:
                    action_r_proposed = get_action_for_state(proposed_r_state, 'right', qpos_numpy[7:14])
                    action_r = (1 - alpha_r) * action_r_committed + alpha_r * action_r_proposed
                else:
                    action_r = action_r_committed

                # Smoothly move toward home during HOLD (including HL HOLD) and switch hold/settle
                if ((plan_r_state == 'HOLD' and not is_transitioning_r) or force_hold_r or settle_after_switch_r > 0):
                    action_r = interp_toward_home(qpos_numpy[7:14], home_pose[7:14])
                    if settle_after_switch_r > 0:
                        settle_after_switch_r -= 1
                
                # Combine
                action = np.concatenate([action_l, action_r])
                
                target_qpos = action

                # Debug mode: run scripted low-level in background and optionally override qpos.
                if scripted_ll_test_mode and debug_ee_env is not None and debug_scripted_policy is not None and debug_ts_ee is not None:
                    try:
                        # Keep EE debug environment object poses aligned with current main env state.
                        sync_debug_ee_objects_from_main(ts, debug_ee_env)
                        debug_ts_ee = type('TS', (object,), {'observation': debug_ee_env.task.get_observation(debug_ee_env.physics)})()
                        debug_scripted_policy.process_command_buffer(debug_ts_ee)
                        ee_action = debug_scripted_policy(debug_ts_ee)
                        debug_ts_ee = debug_ee_env.step(ee_action)
                        use_scripted_override = scripted_ll_override_all_mode or (
                            scripted_ll_coop_only_mode and plan_l_state == 'COOP' and plan_r_state == 'COOP'
                        )
                        if use_scripted_override:
                            target_qpos = np.array(debug_ts_ee.observation['qpos']).copy()
                            # Match replay conversion used in data generation/recording
                            gripper_ctrl = debug_ts_ee.observation.get('gripper_ctrl', None)
                            if gripper_ctrl is not None and len(gripper_ctrl) >= 2:
                                target_qpos[6] = PUPPET_GRIPPER_POSITION_NORMALIZE_FN(gripper_ctrl[0])
                                target_qpos[13] = PUPPET_GRIPPER_POSITION_NORMALIZE_FN(gripper_ctrl[1])
                    except Exception as e:
                        print(f"[Debug Scripted LL] Step failed ({type(e).__name__}): {e}. Falling back to learned action for this step.")
                        try:
                            if debug_ee_env is not None:
                                del debug_ee_env
                            debug_ee_env, debug_scripted_policy, debug_ts_ee = setup_debug_scripted_low_level(
                                ts,
                                getattr(args, 'current_command_sequence', None)
                            )
                            print("[Debug Scripted LL] Reinitialized debug EE runner.")
                        except Exception as reset_e:
                            print(f"[Debug Scripted LL] Reinit failed ({type(reset_e).__name__}): {reset_e}")
                            debug_ee_env, debug_scripted_policy, debug_ts_ee = None, None, None
                
                # --- Safety Clamp REMOVED for debugging ---
                diff = target_qpos - qpos_numpy
                max_diff = np.max(np.abs(diff))
                """
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
                """

                # Subtask smooth-to-home override (same intent as hierarchical_policy.py)
                if subtask_rehome_stage == 'smooth_to_home':
                    max_delta = float(subtask_rehome_interp_delta * 0.8)
                    delta = home_pose - qpos_numpy
                    step = np.clip(delta, -max_delta, max_delta)
                    target_qpos = qpos_numpy + step
                    target_qpos[6] = qpos_numpy[6]
                    target_qpos[13] = qpos_numpy[13]
                    subtask_rehome_settle_left -= 1
                    if subtask_rehome_settle_left <= 0:
                        subtask_rehome_stage = 'none'

                        # Clear temporal/action buffers before next inference phase
                        clear_temporary_states(reset_transition_windows=True)

                        # Reset switch/debounce candidates
                        steps_since_switch_l = 0
                        steps_since_switch_r = 0
                        waiting_home_target_l = None
                        waiting_home_target_r = None
                        switch_candidate_l = None
                        switch_candidate_r = None
                        switch_candidate_count_l = 0
                        switch_candidate_count_r = 0
                        home_pause_count_l = 0
                        home_pause_count_r = 0

                        # Short lock period before normal switching resumes
                        subtask_post_lock_steps = 30

                if subtask_post_lock_steps > 0:
                    subtask_post_lock_steps -= 1

            ts = env.step(target_qpos)
            current_episode_rewards.append(ts.reward)
            
            # --- Magnet Logic (Visual Stacking) ---
            apply_magnet_logic(env.physics, magnetized_pairs, args.color_sequence)
            
            # --- Object Removal Logic (Continuous) ---
            # 1. Track Touches (if needed for debugging, but we mostly care about 'currently in hand')
            new_touches = get_touched_cubes_per_arm(env.physics)
            acc_touched_left.update(new_touches['left'])
            acc_touched_right.update(new_touches['right'])
            
            # 2. Check candidates for removal (Goal or Cushion)
            # NOTE:
            # - Cooperative G/B should NOT be removed individually on goal contact.
            # - Cooperative pair is removed together with delay after BOTH touch goal.
            in_goal = get_cubes_in_goal(env.physics)
            on_cushion = get_cubes_on_cushion(env.physics)
            completed_pairs = getattr(env._task, 'completed_cooperative_pairs', set())

            coop_member_indices = set()
            for pair in completed_pairs:
                try:
                    p0, p1 = pair
                    coop_member_indices.add(p0)
                    coop_member_indices.add(p1)
                except Exception:
                    continue

            def get_color_for_idx(idx):
                try:
                    return color_seq[idx]
                except Exception:
                    return None

            def get_cube_z(idx):
                try:
                    body_id = env.physics.model.name2id(f'cube_{idx}', 'body')
                    return float(env.physics.data.xpos[body_id][2])
                except Exception:
                    return 0.0

            def is_cube_removed_runtime(idx):
                if idx in removed_objects:
                    return True
                try:
                    geom_id = env.physics.model.name2id(f'cube_{idx}', 'geom')
                    alpha = float(env.physics.model.geom_rgba[geom_id, 3])
                except Exception:
                    alpha = 1.0
                z_pos = get_cube_z(idx)
                return (alpha <= 0.01) or (z_pos < -1.0)

            def is_pair_completed(g_idx, b_idx, completed_pair_set):
                return ((g_idx, b_idx) in completed_pair_set) or ((b_idx, g_idx) in completed_pair_set)

            goal_non_coop = {
                idx for idx in in_goal
                if (get_color_for_idx(idx) not in ['g', 'b']) and (idx not in coop_member_indices)
            }
            cushion_non_coop = {
                idx for idx in on_cushion
                if idx not in coop_member_indices
            }
            pending_removal.update(goal_non_coop)
            pending_removal.update(cushion_non_coop)
            forced_pair_removal = set()

            grasped_for_bg = get_grasped_cubes(env.physics)
            touching_for_bg = get_touched_cubes_per_arm(env.physics)
            grasped_now_for_bg = grasped_for_bg['left'] | grasped_for_bg['right']
            ever_grasped_objects.update(grasped_now_for_bg)

            # BG failure handling:
            # Use authoritative cooperative pair sources only (magnetized pairs / locked completed pair)
            # to avoid mixing unrelated G/B cubes from heuristic display targets.
            bg_candidate_pairs = []

            def add_bg_pair(g_idx_local, b_idx_local):
                if g_idx_local is None or b_idx_local is None:
                    return
                if is_cube_removed_runtime(g_idx_local) or is_cube_removed_runtime(b_idx_local):
                    return
                if get_color_for_idx(g_idx_local) != 'g' or get_color_for_idx(b_idx_local) != 'b':
                    return
                pair_local = (g_idx_local, b_idx_local)
                pair_local_rev = (b_idx_local, g_idx_local)
                if pair_local not in bg_candidate_pairs and pair_local_rev not in bg_candidate_pairs:
                    bg_candidate_pairs.append(pair_local)

            # 1) Completed/locked cooperative pair (if any)
            if coop_display_lock_pair is not None:
                g_lock, b_lock = coop_display_lock_pair
                if get_color_for_idx(g_lock) == 'b' and get_color_for_idx(b_lock) == 'g':
                    g_lock, b_lock = b_lock, g_lock
                add_bg_pair(g_lock, b_lock)

            # 2) Magnetized pairs are the ground truth for current G-B coupling.
            for g_idx_mag, data_mag in magnetized_pairs.items():
                b_idx_mag = data_mag.get('blue_idx', None)
                add_bg_pair(g_idx_mag, b_idx_mag)

            # 2.5) Safe fallback: if coop shadow currently shows exactly one G/B pair,
            # use that exact pair (no heuristic mixing across different indices).
            if 'final_target_indices_c' in locals() and len(final_target_indices_c) == 2:
                p0, p1 = final_target_indices_c[0], final_target_indices_c[1]
                c0 = get_color_for_idx(p0)
                c1 = get_color_for_idx(p1)
                if c0 == 'g' and c1 == 'b':
                    add_bg_pair(p0, p1)
                elif c0 == 'b' and c1 == 'g':
                    add_bg_pair(p1, p0)

            # 3) Keep active_coop_pair only when it matches an actual magnetized pair.
            if active_coop_pair is not None:
                g_act, b_act = active_coop_pair
                if get_color_for_idx(g_act) == 'b' and get_color_for_idx(b_act) == 'g':
                    g_act, b_act = b_act, g_act
                if g_act in magnetized_pairs and magnetized_pairs[g_act].get('blue_idx', None) == b_act:
                    add_bg_pair(g_act, b_act)

            if args.debug_bg_removal_logs:
                should_log_bg = (
                    (t % args.debug_bg_log_interval == 0)
                    or bool(on_cushion)
                    or bool(in_goal)
                    or bool(bg_candidate_pairs)
                    or bool(forced_pair_removal)
                )
                if should_log_bg:
                    print(
                        f"[Step {t}] BG debug | on_cushion={sorted(list(on_cushion))} "
                        f"in_goal={sorted(list(in_goal))} bg_pairs={bg_candidate_pairs} "
                        f"magnet_pairs={[(g, d.get('blue_idx', None)) for g, d in magnetized_pairs.items()]} "
                        f"active_pair={active_coop_pair} lock_pair={coop_display_lock_pair}"
                    )

            ever_contacted_objects = set(ever_grasped_objects)
            ever_contacted_objects.update(acc_touched_left)
            ever_contacted_objects.update(acc_touched_right)

            active_pair_keys = set()

            for pair in bg_candidate_pairs:
                try:
                    g_idx, b_idx = pair
                except Exception:
                    continue

                if get_color_for_idx(g_idx) == 'b' and get_color_for_idx(b_idx) == 'g':
                    g_idx, b_idx = b_idx, g_idx

                if get_color_for_idx(g_idx) != 'g' or get_color_for_idx(b_idx) != 'b':
                    continue

                if is_cube_removed_runtime(g_idx) or is_cube_removed_runtime(b_idx):
                    coop_pair_release_streak.pop((g_idx, b_idx), None)
                    coop_pair_release_streak.pop((b_idx, g_idx), None)
                    continue

                pair_key = (g_idx, b_idx)
                active_pair_keys.add(pair_key)

                if is_pair_completed(g_idx, b_idx, completed_pairs):
                    coop_pair_release_streak.pop(pair_key, None)
                    continue

                fail_goal = (g_idx in in_goal) or (b_idx in in_goal)
                fail_cushion = (g_idx in on_cushion) or (b_idx in on_cushion)
                fail_fall = (get_cube_z(g_idx) < -0.1) or (get_cube_z(b_idx) < -0.1)

                # Immediate BG-fail conditions should not depend on grasp/touch history.
                if fail_goal or fail_cushion or fail_fall:
                    reason_tokens = []
                    if fail_goal:
                        reason_tokens.append('goal')
                    if fail_cushion:
                        reason_tokens.append('cushion')
                    if fail_fall:
                        reason_tokens.append('fall')
                    reason = '/'.join(reason_tokens)
                    print(f"[Step {t}] BG fail pair removal scheduled: G{g_idx}+B{b_idx} reason={reason}")
                    forced_pair_removal.update([g_idx, b_idx])
                    coop_pair_release_streak.pop(pair_key, None)
                    continue

                ever_contacted_pair = (g_idx in ever_contacted_objects) or (b_idx in ever_contacted_objects)
                if not ever_contacted_pair:
                    coop_pair_release_streak.pop(pair_key, None)
                    continue

                touching_pair_now = (
                    (g_idx in touching_for_bg['left']) or (b_idx in touching_for_bg['left']) or
                    (g_idx in touching_for_bg['right']) or (b_idx in touching_for_bg['right'])
                )
                if touching_pair_now:
                    coop_pair_release_streak[pair_key] = 0
                else:
                    coop_pair_release_streak[pair_key] = coop_pair_release_streak.get(pair_key, 0) + 1

                if coop_pair_release_streak.get(pair_key, 0) >= 20:
                    print(f"[Step {t}] BG fail pair removal scheduled: G{g_idx}+B{b_idx} reason=release20")
                    forced_pair_removal.update([g_idx, b_idx])
                    continue

            for stale_pair in list(coop_pair_release_streak.keys()):
                if stale_pair not in active_pair_keys:
                    del coop_pair_release_streak[stale_pair]

            # Cooperative pair delayed removal latch:
            # start timer when ACTIVE pair is marked completed by env reward logic.
            for pair in completed_pairs:
                try:
                    g_idx, b_idx = pair
                except Exception:
                    continue

                # Safety for unexpected ordering
                c0 = get_color_for_idx(g_idx)
                c1 = get_color_for_idx(b_idx)
                if c0 == 'b' and c1 == 'g':
                    g_idx, b_idx = b_idx, g_idx

                if get_color_for_idx(g_idx) != 'g' or get_color_for_idx(b_idx) != 'b':
                    continue

                pair_key = (g_idx, b_idx)
                pair_key_rev = (b_idx, g_idx)
                is_active_pair = (
                    active_coop_pair is not None and
                    (active_coop_pair == pair_key or active_coop_pair == pair_key_rev)
                )
                if is_active_pair and pair_key not in coop_goal_pair_start_step and pair_key_rev not in coop_goal_pair_start_step:
                    coop_goal_pair_start_step[pair_key] = t
                    coop_display_lock_pair = pair_key
                    print(f"[Step {t}] Coop pair completion latched: G{g_idx}+B{b_idx}")
            
            # 3. Remove if NOT protected (touched/grasped/nearby)
            if pending_removal or coop_goal_pair_start_step or forced_pair_removal:
                 grasped = get_grasped_cubes(env.physics)
                 grasped_now = grasped['left'] | grasped['right']
                 ever_grasped_objects.update(grasped_now)
                 nearby = get_proximity_cubes(env.physics)
                 currently_touching = get_touched_cubes_per_arm(env.physics)

                 protected_strict = (grasped['left'] | grasped['right'] |
                                     currently_touching['left'] | currently_touching['right'])
                 protected_any = (protected_strict | nearby)

                 # Strict policy with hysteresis:
                 # remove previously grasped non-coop cubes only after consecutive
                 # "released + unprotected" frames, to avoid one-frame grasp flicker.
                 RELEASE_REMOVE_HYSTERESIS = 8
                 released_grasped = set()
                 for idx in ever_grasped_objects:
                     if idx in removed_objects:
                         continue
                     if get_color_for_idx(idx) in ['g', 'b']:
                         continue

                     released_and_unprotected = (idx not in grasped_now) and (idx not in protected_any)
                     if released_and_unprotected:
                         grasp_release_streak[idx] = grasp_release_streak.get(idx, 0) + 1
                     else:
                         grasp_release_streak[idx] = 0

                     if grasp_release_streak.get(idx, 0) >= RELEASE_REMOVE_HYSTERESIS:
                         released_grasped.add(idx)

                 # Keep streak dict compact.
                 for idx in list(grasp_release_streak.keys()):
                     if idx in removed_objects or idx not in ever_grasped_objects:
                         del grasp_release_streak[idx]

                 # Hard guard: never single-remove cooperative color cubes.
                 pending_removal = {idx for idx in pending_removal if get_color_for_idx(idx) not in ['g', 'b']}

                 # Standard single-object removal (non-coop goal + cushion)
                 # Do not block by mere proximity; only active grasp/touch should protect.
                 to_remove = set(pending_removal - protected_strict)
                 to_remove.update(released_grasped)
                 to_remove.update(forced_pair_removal)
                 to_remove = {idx for idx in to_remove if not is_cube_removed_runtime(idx)}

                 # Cooperative pair delayed joint removal
                 REMOVAL_DELAY_STEPS = int(1.0 / DT)
                 for pair_key, start_t in list(coop_goal_pair_start_step.items()):
                     g_idx, b_idx = pair_key
                     elapsed = t - start_t
                     if elapsed > REMOVAL_DELAY_STEPS:
                         # Remove pair only when BOTH are safe to remove.
                         if g_idx not in protected_any and b_idx not in protected_any:
                             to_remove.update([g_idx, b_idx])
                             del coop_goal_pair_start_step[pair_key]

                 if to_remove:
                     print(f"[Step {t}] Auto-Removing objects: {to_remove}")
                     remove_cubes(env.physics, list(to_remove))
                     removed_objects.update(to_remove)
                     ever_grasped_objects -= to_remove
                     pending_removal -= to_remove
                     for idx in to_remove:
                         grasp_release_streak.pop(idx, None)

                     for pair_key in list(coop_pair_release_streak.keys()):
                         if pair_key[0] in to_remove or pair_key[1] in to_remove:
                             del coop_pair_release_streak[pair_key]

                     # Drop stale pair timers that include removed cubes
                     for pair_key in list(coop_goal_pair_start_step.keys()):
                         if pair_key[0] in to_remove or pair_key[1] in to_remove:
                             del coop_goal_pair_start_step[pair_key]

                     # Release display lock once locked pair is removed.
                     if coop_display_lock_pair is not None:
                         if coop_display_lock_pair[0] in to_remove or coop_display_lock_pair[1] in to_remove:
                             coop_display_lock_pair = None
                             active_coop_pair = None
                     
                     # Update Display Logic: Trigger cooldown and remove from current display list
                     last_removal_step = t
                     # Re-cast to lists to allow modification if they are tuples/etc, though initiated as lists
                     current_target_indices_i = [idx for idx in current_target_indices_i if idx not in to_remove]
                     current_target_indices_c = [idx for idx in current_target_indices_c if idx not in to_remove]

                     # Subtask rehome trigger: after successful removal, return home then resume
                     subtask_rehome_trigger_pending = True
            
            step_in_chunk += 1
            if current_action_chunk_dual is not None:
                step_in_chunk_dual += 1
            if current_action_chunk_left is not None:
                step_in_chunk_left += 1
            if current_action_chunk_right is not None:
                step_in_chunk_right += 1
            t += 1
            
            # Reset logic matching imitate_episodes num_rollouts loop
            if t >= max_timesteps:
                print("Episode finished. Resetting...")
                
                # Calculate Episode Status
                max_possible_reward = env._task.max_reward
                final_reward = current_episode_rewards[-1] if current_episode_rewards else 0
                is_success = (final_reward >= max_possible_reward)

                # Save Video
                if args.save_video and len(video_frames) > 0:
                     video_dir = args.save_video if isinstance(args.save_video, str) else 'videos'
                     if not os.path.exists(video_dir):
                         os.makedirs(video_dir)
                         
                     # Result recording for filename
                     status_str = "success" if is_success else "fail"
                     video_path = os.path.join(video_dir, f'eval_ep{episode_count}_{status_str}_r{final_reward}.mp4')
                     
                     # Detect shape from first frame
                     target_h, target_w, _ = video_frames[0].shape
                     fps = 30
                     # Use mp4v for mp4
                     out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (target_w, target_h))
                     for frame in video_frames:
                         # RGB to BGR for CV2
                         frame_bgr = frame[:, :, [2, 1, 0]]
                         out.write(frame_bgr)
                     out.release()
                     print(f"Saved video to {video_path}")
                
                video_frames = [] # Clear for next episode

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
                ever_grasped_objects.clear()
                removed_objects.clear()
                coop_goal_pair_start_step.clear()
                coop_display_lock_pair = None
                active_coop_pair = None
                coop_pair_release_streak.clear()
                grasp_release_streak.clear()
                magnetized_pairs.clear() # FIX: Clear magnet state!
                
                # Reset State Tracking
                prev_plan_l_state = None
                prev_plan_r_state = None
                last_active_l_state = None
                last_active_r_state = None
                state_history_l.clear()
                state_history_r.clear()
                hl_mode_l = STATE_INDEP
                hl_mode_r = STATE_INDEP
                last_hl_update_step_l = -1
                last_hl_update_step_r = -1

                # Reset debounce / committed mode state
                committed_plan_l_state = 'INDEP'
                committed_plan_r_state = 'INDEP'
                steps_since_switch_l = 0
                steps_since_switch_r = 0

                # Reset hold/home counters
                consecutive_at_home_l = 0
                consecutive_at_home_r = 0

                # Reset transition windows
                transition_window_active_l = 0
                transition_window_active_r = 0

                # Reset display target update state
                last_removal_step = -100
                current_target_indices_i = []
                current_target_indices_c = []

                switch_candidate_l = None
                switch_candidate_r = None
                switch_candidate_count_l = 0
                switch_candidate_count_r = 0
                waiting_home_target_l = None
                waiting_home_target_r = None
                home_pause_count_l = 0
                home_pause_count_r = 0
                settle_after_switch_l = 0
                settle_after_switch_r = 0
                left_waiting_indep_target = False
                left_resume_reinfer_pending = False
                right_waiting_indep_target = False
                right_resume_reinfer_pending = False

                # Reset subtask rehome state machine
                subtask_rehome_stage = 'none'
                subtask_rehome_trigger_pending = False
                subtask_rehome_settle_left = 0
                subtask_post_lock_steps = 0

                # Save Stats to CSV
                if args.save_stats_path:
                    # Get Task Status
                    try:
                        task = env._task
                        col_seq = task.color_sequence if task.color_sequence else ['?']*10
                        
                        # Prepare Row Data
                        row = {
                            'Episode': episode_count,
                            'Total_Reward': episode_return,
                            'Is_Success': is_success
                        }
                        
                        # Verify using internal sets
                        # Note: 'task' object might be wrapped. verify env access.
                        # env._task should differ based on make_sim_env implementation details, 
                        # but in piper_sim_env.py it is directly accessible.
                        
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

                # Save HITL Data if collected
                if args.save_hitl_data_path and len(hitl_images) > 0:
                    try:
                         # Convert to numpy
                         # Images: list of (H, W, C) -> (N, H, W, C) ? Or (N, H, W, C)?
                         # Sim render returns (H, W, C)
                         imgs_np = np.array(hitl_images) 
                         lbls_l_np = np.array(hitl_labels_l)
                         lbls_r_np = np.array(hitl_labels_r)
                         
                         save_hitl_data(args.save_hitl_data_path, episode_count, imgs_np, lbls_l_np, lbls_r_np)
                    except Exception as e:
                         print(f"Error saving HITL data: {e}")

                if episode_count >= args.num_rollouts:
                    break

                # Cleanup / Preparation for next episode
                if not apply_next_episode_sequence(episode_count):
                    break
                reset_magnet_logic(env.physics, args.color_sequence)
                ts = reset_with_new_pose()
                debug_ee_env, debug_scripted_policy, debug_ts_ee = setup_debug_scripted_low_level(ts, getattr(args, 'current_command_sequence', None))
                current_hl_oracle_schedule = _select_oracle_schedule(episode_count, getattr(args, 'current_sequence_row', None)) if args.hl_oracle_metadata_csv else None
                if current_hl_oracle_schedule is not None:
                    print(f"[HL Oracle] Enabled for episode {episode_count} (sequence_row={getattr(args, 'current_sequence_row', None)})")
                t = 0
                step_in_chunk = 0
                step_in_chunk_dual = 0
                step_in_chunk_left = 0
                step_in_chunk_right = 0
                current_action_chunk_dual = None
                current_action_chunk_left = None
                current_action_chunk_right = None
                
                # Reset temporary states/buffers
                clear_temporary_states(reset_transition_windows=False)
                
                # Reset HITL Buffers
                hitl_mode_l = 'AUTO'
                hitl_mode_r = 'AUTO'
                hitl_images = []
                hitl_labels_l = []
                hitl_labels_r = []
                    
        # Summary
        # Save any partial HITL data if loop was broken early
        if args.save_hitl_data_path and len(hitl_images) > 0:
            try:
                 print(f"Saving partial HITL data ({len(hitl_images)} frames)...")
                 imgs_np = np.array(hitl_images) 
                 lbls_l_np = np.array(hitl_labels_l)
                 lbls_r_np = np.array(hitl_labels_r)
                 
                 # Use a distinct suffix for partial data
                 save_hitl_data(args.save_hitl_data_path, f"{episode_count}_partial", imgs_np, lbls_l_np, lbls_r_np)
            except Exception as e:
                 print(f"Error saving partial HITL data: {e}")

        if len(episode_returns) > 0:
            avg_return = np.mean(episode_returns)
            print(f"\nEvaluation Finished.")
            print(f"Average Return: {avg_return:.2f}")
        else:
            print(f"\nEvaluation Finished (No complete episodes).")
            print(f"Average Return: N/A")

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
    parser.add_argument('--save_video', nargs='?', const='videos', type=str, help='Save execution video to mp4 (optional path, default "videos")')
    parser.add_argument('--save_stats_path', action='store', type=str, help='Path to save episode statistics CSV')
    parser.add_argument('--num_rollouts', action='store', type=int, default=1, help='Number of evaluation episodes')
    parser.add_argument('--color_sequence', action='store', type=str, default=None, help='Color sequence (e.g. rrgbrrgbrr)')
    parser.add_argument('--episode_len', action='store', type=int, default=None, help='Override task-specific episode length')
    parser.add_argument('--inherit_temporal_buffer', action='store_true', help='Inherit temporal aggregation buffer on switch to prevent jerk')
    parser.add_argument('--warmup_steps', action='store', type=int, default=0, help='Number of steps to run independent policy in background before switch')
    parser.add_argument('--no_temporal_agg', action='store_true', help='Disable temporal aggregation')
    parser.add_argument('--temporal_agg_transition_only', action='store_true', help='Enable temporal ensembling only around transitions')
    parser.add_argument('--temporal_agg_window', action='store', type=int, default=50, help='Number of steps around transition to enable temporal agg')
    parser.add_argument('--hl_update_at_home_only', action='store_true', help='Update high-level classifier only when both arms are at home')
    parser.add_argument('--hl_update_interval', action='store', type=int, default=50, help='Step interval for high-level updates when home-only mode is enabled')
    parser.add_argument('--hl_home_threshold', action='store', type=float, default=0.25, help='Arm joint threshold for hierarchical-style high-level home check')
    parser.add_argument('--hl_home_gripper_threshold', action='store', type=float, default=0.8, help='Gripper-open threshold for hierarchical-style high-level home check')
    parser.add_argument('--hl_update_log_interval', action='store', type=int, default=50, help='Step interval for logging non-fired high-level updates in home-only mode')
    parser.add_argument('--debug_hl_logs', action='store_true', help='Enable verbose high-level/update/home-refresh debug logs')
    parser.add_argument('--switch_guard_steps', action='store', type=int, default=8, help='Stable high-level prediction steps required before requesting mode switch')
    parser.add_argument('--switch_home_pause_steps', action='store', type=int, default=15, help='Pause steps at home before activating next policy')
    parser.add_argument('--switch_home_settle_steps', action='store', type=int, default=6, help='Additional steps to keep exact home pose after switch activation')
    parser.add_argument('--switch_home_interp_max_delta', action='store', type=float, default=0.04, help='Max per-step joint delta when interpolating toward home during switch hold/settle')
    parser.add_argument('--switch_home_threshold', action='store', type=float, default=0.12, help='Home detection threshold for gated switching')
    parser.add_argument('--x_shift', action='store', type=float, default=0.0, help='Shift all objects along X-axis')
    parser.add_argument('--x_shift_start_idx', action='store', type=int, default=0, help='Start index for applying x-shift (0-indexed)')
    
    # HITL Arguments
    parser.add_argument('--save_hitl_data_path', action='store', type=str, help='Path to save HITL correction data (HDF5)')
    parser.add_argument('--seed', action='store', type=int, default=1000, help='Random seed for reproducibility')
    
    parser.add_argument('--mask_images', action='store_true', help="Enable masking of objects based on policy type")
    parser.add_argument('--temporal_agg_k', action='store', type=float, default=0.01, help='Exponential decay factor k for Temporal Aggregation')
    parser.add_argument('--sequence_file', action='store', type=str, help='Path to CSV sequence file (Optional override for color_sequence)', default=None)
    parser.add_argument('--debug_scripted_low_level', action='store_true', help='Debug mode: override low-level actions with data-generation scripted policy (InteractivePolicy in EE env)')
    parser.add_argument('--unit_test_scripted_low_level', action='store_true', help='Unit-test mode: run low-level using the data-generation scripted policy (alias of --debug_scripted_low_level)')
    parser.add_argument('--debug_scripted_coop_low_level_only', action='store_true', help='Debug mode: override only COOP/COOP low-level steps with scripted policy; keep independent low-level learned')
    parser.add_argument('--hl_oracle_metadata_csv', action='store', type=str, default=None,
                        help='Optional metadata CSV (e.g., dryrun_sequence_metadata output) to replay handcrafted high-level mode transitions')
    parser.add_argument('--hl_oracle_future_steps', action='store', type=int, default=50,
                        help='When using hl_oracle_metadata_csv, query oracle mode at t+N (default: 50) to match future-predicting high-level timing')
    parser.add_argument('--debug_bg_removal_logs', action='store_true', help='Enable debug logs for BG pair candidate/removal conditions')
    parser.add_argument('--debug_bg_log_interval', action='store', type=int, default=50, help='Step interval for periodic BG debug logs')

    args = parser.parse_args()
    main(args)
