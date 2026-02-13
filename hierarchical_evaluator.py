import os
import sys
import torch
import torch.nn as nn
import numpy as np
import cv2
import json
import pickle
import argparse
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from copy import deepcopy

# Add detr to path
cwd = os.getcwd()
sys.path.append(os.path.join(cwd, 'detr'))

# Project imports
from piper_constants import DT, START_ARM_POSE, SIM_TASK_CONFIGS
from piper_sim_env import make_sim_env, MANYCUBES_COLORS
from policy import ACTPolicy
from utils import set_seed

# Transformations for High-level Classifier
CLS_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class MultiHeadResNet18(nn.Module):
    def __init__(self, num_modes=2, num_objs=7):
        super(MultiHeadResNet18, self).__init__()
        import torchvision.models as models
        self.backbone = models.resnet18(pretrained=False)
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.fc_l_mode = nn.Linear(num_ftrs, num_modes)
        self.fc_r_mode = nn.Linear(num_ftrs, num_modes)
        self.fc_l_obj = nn.Linear(num_ftrs, num_objs)
        self.fc_r_obj = nn.Linear(num_ftrs, num_objs)
        
    def forward(self, x):
        features = self.backbone(x)
        return self.fc_l_mode(features), self.fc_r_mode(features), \
               self.fc_l_obj(features), self.fc_r_obj(features)

def load_multihead_classifier(model_path):
    model = MultiHeadResNet18()
    model.load_state_dict(torch.load(model_path))
    model.cuda()
    model.eval()
    return model

def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy

def load_policy_and_stats(ckpt_dir, policy_class, args, override_state_dim=None, override_arm=None):
    state_dim = 14
    if override_state_dim:
        state_dim = override_state_dim
        
    camera_names = ['top'] 
    lr_backbone = 1e-5
    backbone = 'resnet18'
    
    policy_config = {
        'lr': 1e-5,
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
        ckpt_path = ckpt_dir
        parent_dir = os.path.dirname(ckpt_dir)
        stats_path = os.path.join(parent_dir, 'dataset_stats.pkl')
    else:
        ckpt_path = os.path.join(ckpt_dir, 'policy_best.ckpt')
        stats_path = os.path.join(ckpt_dir, 'dataset_stats.pkl')

    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)
    
    policy = make_policy(policy_class, policy_config)
    loaded_state_dict = torch.load(ckpt_path)
    if 'model_state_dict' in loaded_state_dict:
        loaded_state_dict = loaded_state_dict['model_state_dict']
    
    policy.load_state_dict(loaded_state_dict)
    policy.cuda()
    policy.eval()
    return policy, stats

def apply_torch_rgb_mask(images, strip_width=40, arm='right'):
    """Applies RGB mask to a strip of the images (B, C, H, W)."""
    B, C, H, W = images.shape
    if arm == 'left':
        strip_start, strip_end = W - strip_width, W
    else:
        strip_start, strip_end = 0, strip_width
        
    strip = images[:, :, :, strip_start:strip_end]
    t_100, t_150 = 100.0/255.0, 150.0/255.0
    r, g, b = strip[:, 0, :, :], strip[:, 1, :, :], strip[:, 2, :, :]
    mask = ((r > t_100) & (g < t_100) & (b < t_100)) | \
           ((g > t_100) & (r < t_100) & (b < t_100)) | \
           ((b > t_150) & (r < t_100) & (g < t_100))
    
    combined_mask = mask.unsqueeze(1).repeat(1, C, 1, 1).float()
    masked_strip = strip * combined_mask
    outputs = images.clone()
    outputs[:, :, :, strip_start:strip_end] = masked_strip
    return outputs

def is_at_home(current_qpos, threshold=0.25, gripper_threshold=0.8):
    # 1. Check arm joints
    home_l = np.array(START_ARM_POSE[:6])
    home_r = np.array(START_ARM_POSE[8:14])
    diff_l = np.max(np.abs(current_qpos[:6] - home_l))
    diff_r = np.max(np.abs(current_qpos[7:13] - home_r))
    
    arms_at_home = diff_l < threshold and diff_r < threshold

    # 2. Check grippers (0: Close, 1: Open)
    # Must be open to consider "task finished" and ready for new mode
    # Indices: Left=6, Right=13
    left_open = current_qpos[6] > gripper_threshold
    right_open = current_qpos[13] > gripper_threshold
    
    grippers_open = left_open and right_open

    return arms_at_home and grippers_open, diff_l, diff_r

def sync_envs(main_physics, shadow_physics):
    """Sync robot/object state AND visual properties (geom_rgba) from main to shadow."""
    # 1. State Sync
    shadow_physics.data.qpos[:] = main_physics.data.qpos[:]
    shadow_physics.data.qvel[:] = main_physics.data.qvel[:]
    
    # 2. Visual Sync (geom_rgba determines cube colors)
    shadow_physics.model.geom_rgba[:] = main_physics.model.geom_rgba[:]
    
    shadow_physics.forward()

def remove_cubes(physics, indices):
    """Teleport cubes to z=-10 to effectively remove them from simulation."""
    if not indices: return
    for i in indices:
        try:
            name = f'cube_{i}'
            # 1. Teleport Joint
            joint_name = f'cube_{i}_joint'
            addr = physics.model.name2id(joint_name, 'joint')
            qpos_adr = physics.model.jnt_qposadr[addr]
            physics.data.qpos[qpos_adr + 0] = 10.0 + i # X far
            physics.data.qpos[qpos_adr + 1] = 10.0 + i # Y far
            physics.data.qpos[qpos_adr + 2] = -10.0    # Z under
            
            # 2. Set Alpha to 0
            geom_id = physics.model.name2id(name, 'geom')
            physics.model.geom_rgba[geom_id, 3] = 0.0
        except: pass
    physics.forward()

def get_goal_object_indices(physics):
    """Return list of cube indices currently touching the goal_plate."""
    indices = []
    # Get all contact pairs
    all_contacts = set()
    for i in range(physics.data.ncon):
        id1, id2 = physics.data.contact[i].geom1, physics.data.contact[i].geom2
        name1, name2 = physics.model.id2name(id1, 'geom'), physics.model.id2name(id2, 'geom')
        if name1 and name2:
            all_contacts.add((name1, name2))
            all_contacts.add((name2, name1))
            
    for i in range(10):
        cube_name = f'cube_{i}'
        if ('goal_plate', cube_name) in all_contacts:
            indices.append(i)
        else:
            try:
                bid = physics.model.name2id(cube_name, 'body')
                pos = physics.data.xpos[bid]
                # Debug print for objects near goal
                if abs(pos[0]) < 0.2 and 0.0 < pos[1] < 0.2:
                    # print(f"  [Debug] {cube_name} at X={pos[0]:.3f}, Y={pos[1]:.3f}, Z={pos[2]:.3f}") # Disabled to avoid spam, but keep for easy toggle
                    pass
                
                if abs(pos[0]) < 0.1 and 0.04 < pos[1] < 0.14 and pos[2] < 0.15: # Raised Z for stacking
                    indices.append(i)
            except: pass
    return list(set(indices))

def get_heuristic_targets(physics, spatial_map, color_sequence):
    """
    Determine target indices based on mode requirements and spatial heuristics.
    Returns: (target_indices_i, target_indices_c)
    """
    target_indices_i = []
    target_indices_c = []
    
    def safe_get_color(idx):
        try: return color_sequence[idx]
        except: return None

    # 1. Cooperative Shadow (target_indices_c)
    coop_candidates = []
    for label, c_idx in spatial_map.items():
        try:
            bid = physics.model.name2id(f'cube_{c_idx}', 'body')
            x_pos = physics.data.xpos[bid][0]
            if x_pos < 0.3:
                color = safe_get_color(c_idx)
                if color in ['g', 'b']:
                    coop_candidates.append((x_pos, c_idx, color))
        except: pass
    
    if coop_candidates:
        rightmost_g = max([c for c in coop_candidates if c[2] == 'g'], key=lambda x: x[0], default=None)
        rightmost_b = max([c for c in coop_candidates if c[2] == 'b'], key=lambda x: x[0], default=None)
        if rightmost_g: target_indices_c.append(rightmost_g[1])
        if rightmost_b: target_indices_c.append(rightmost_b[1])

    # 2. Independent Shadow (target_indices_i)
    rh_candidates = []
    lh_candidates = []
    for label, c_idx in spatial_map.items():
        try:
            bid = physics.model.name2id(f'cube_{c_idx}', 'body')
            x_pos = physics.data.xpos[bid][0]
            color = safe_get_color(c_idx)
            if color == 'r':
                if 0 < x_pos < 0.3:
                    rh_candidates.append((x_pos, c_idx))
                if x_pos < 0.1:
                    lh_candidates.append((x_pos, c_idx))
        except: pass
    
    if rh_candidates:
        target_indices_i.append(max(rh_candidates, key=lambda x: x[0])[1])
    if lh_candidates:
        target_indices_i.append(max(lh_candidates, key=lambda x: x[0])[1])

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
        except: pass
        
        # 2. Move qpos if hiding (independent of geom success)
        if i not in target_set:
            try:
                joint_name = f'cube_{i}_joint'
                addr = physics.model.name2id(joint_name, 'joint')
                qpos_adr = physics.model.jnt_qposadr[addr]
                physics.data.qpos[qpos_adr + 2] = -5.0 # Well below table
            except: pass
            
    physics.forward()

def get_spatial_object_map(physics):
    """Map L0, R0 etc to cube indices based on x-coordinate relative to x=0."""
    cubes = []
    for i in range(10):
        try:
            name = f'cube_{i}'
            bid = physics.model.name2id(name, 'body')
            x = physics.data.xpos[bid][0]
            cubes.append({'id': i, 'x': x})
        except: pass
    
    # Sort symmetrically (Matches extract_transitions.py)
    # L0 is closest to center (highest X), L1 is further left.
    left_cubes = sorted([c for c in cubes if c['x'] < 0], key=lambda c: c['x'], reverse=True)
    # R0 is closest to center (lowest X), R1 is further right.
    right_cubes = sorted([c for c in cubes if c['x'] >= 0], key=lambda c: c['x'])
    
    obj_map = {}
    for i, c in enumerate(left_cubes): obj_map[f'L{i}'] = c['id']
    for i, c in enumerate(right_cubes): obj_map[f'R{i}'] = c['id']
    
    return obj_map

def is_success(physics):
    """Check if any cube is in the goal zone [x:-0.1~0.1, y:0.04~0.14]."""
    success_count = 0
    for i in range(10):
        try:
            name = f'cube_{i}'
            bid = physics.model.name2id(name, 'body')
            pos = physics.data.xpos[bid]
            # Goal zone check
            if abs(pos[0]) < 0.1 and 0.04 < pos[1] < 0.14 and pos[2] < 0.05:
                success_count += 1
        except: pass
    return success_count

def main(args):
    set_seed(args.seed)
    
    # Load Models
    print("Loading Multi-head High-level Classifier...")
    cls_model = load_multihead_classifier(args.cls_ckpt)
    
    print("Loading Low-level Policies...")
    policy_c, stats_c = load_policy_and_stats(args.ckpt_c, 'ACT', args, override_state_dim=14)
    policy_l, stats_l = load_policy_and_stats(args.ckpt_l, 'ACT', args, override_state_dim=7, override_arm='left')
    policy_r, stats_r = load_policy_and_stats(args.ckpt_r, 'ACT', args, override_state_dim=7, override_arm='right')
    
    # Envs
    task_config = SIM_TASK_CONFIGS[args.task_name]
    time_limit = (task_config['episode_len'] + 200) * DT
    
    if args.color_sequence:
        MANYCUBES_COLORS[0] = list(args.color_sequence.lower())
    else:
        print("Warning: No color sequence provided. Using default 'rgbrgbrgbr'.")
        default_seq = "rgbrgbrgbr"
        MANYCUBES_COLORS[0] = list(default_seq)
        args.color_sequence = default_seq

    env_main = make_sim_env(args.task_name, time_limit=time_limit)
    env_shadow_c = make_sim_env(args.task_name, time_limit=time_limit)
    env_shadow_i = make_sim_env(args.task_name, time_limit=time_limit)
    
    ts = env_main.reset()

    post_process_c = lambda a: a * stats_c['action_std'] + stats_c['action_mean']
    post_process_l = lambda a: a * stats_l['action_std'] + stats_l['action_mean']
    post_process_r = lambda a: a * stats_r['action_std'] + stats_r['action_mean']

    if args.onscreen_render:
        plt.ion()
        fig, (ax_main, ax_c, ax_i) = plt.subplots(1, 3, figsize=(15, 5))
        plt_main = ax_main.imshow(np.zeros((240, 320, 3), dtype=np.uint8))
        plt_c = ax_c.imshow(np.zeros((240, 320, 3), dtype=np.uint8))
        plt_i = ax_i.imshow(np.zeros((240, 320, 3), dtype=np.uint8))

    l_mode, r_mode = 1, 1 
    removed_objects = set()
    max_steps = args.episode_len or task_config['episode_len']

    # Ensembling parameters
    chunk_size = args.chunk_size
    all_time_actions_c = np.zeros([max_steps, max_steps + chunk_size, 14])
    all_time_actions_l = np.zeros([max_steps, max_steps + chunk_size, 7])
    all_time_actions_r = np.zeros([max_steps, max_steps + chunk_size, 7])
    
    k = 0.01 # Exp weighting factor
    exp_weights = np.exp(-k * np.arange(chunk_size))
    exp_weights = exp_weights / exp_weights.sum()

    print(f"Starting loop for {max_steps} steps...")
    for t in range(max_steps):
        qpos_main = ts.observation['qpos']
        
        # 1. High-level check
        at_home, dl, dr = is_at_home(qpos_main)
        if at_home and t % 50 == 0:
            full_img = ts.observation['images']['top']
            H, W, _ = full_img.shape
            # Match extract_transitions.py: crop_y1, crop_y2 = H//4, H//2
            img = full_img[H//4 : H//2, :].copy()
            img_t = CLS_TRANSFORM(Image.fromarray(img)).unsqueeze(0).cuda()
            
            with torch.no_grad():
                out_lm, out_rm, out_lo, out_ro = cls_model(img_t)
                l_mode = torch.argmax(out_lm, 1).item()
                r_mode = torch.argmax(out_rm, 1).item()
            
            print(f"[{t}] High-level Pred -> L_Mode:{l_mode}, R_Mode:{r_mode}")
            
            # Debug Mapping (Only log when querying)
            sm = get_spatial_object_map(env_main.physics)
            print(f"  [Spatial Mapping] {len(sm)} objects detected:")
            for label, cid in sm.items():
                # Get body ID safely
                try:
                    bid = env_main.physics.model.name2id(f'cube_{cid}', 'body')
                    x_pos = env_main.physics.data.xpos[bid][0]
                    print(f"    - {label}: cube_{cid} (X={x_pos:.3f})")
                except: pass

        # 2. Sync, Detect Gold, and Filter
        goal_indices = get_goal_object_indices(env_main.physics)
        for idx in goal_indices:
            if idx not in removed_objects:
                print(f"[Step {t}] SUCCESS: Object {idx} in Goal. Removing.")
                removed_objects.add(idx)
        
        # Apply permanent removal to main
        remove_cubes(env_main.physics, removed_objects)

        sync_envs(env_main.physics, env_shadow_c.physics)
        sync_envs(env_main.physics, env_shadow_i.physics)
        
        # 3. Apply Heuristic Masking
        spatial_map = get_spatial_object_map(env_main.physics)
        color_seq = getattr(env_main.task, 'color_sequence', MANYCUBES_COLORS[0])
        target_indices_i, target_indices_c = get_heuristic_targets(env_main.physics, spatial_map, color_seq)
        
        if t % 100 == 0:
            print(f"  [Heuristic Targets] Step {t}: Indep={target_indices_i}, Coop={target_indices_c}")
            
        hide_objects(env_shadow_i.physics, target_indices_i, "Indep")
        hide_objects(env_shadow_c.physics, list(set(target_indices_c)), "Coop")
            
        # 3. Action Generation with Temporal Ensembling
        
        # --- Policy Execution ---
        # Execute policies based on current modes. 
        # Stores predictions in all_time_actions buffers.
        
        # CASE 1: Independent Policies (Low-level)
        # If either arm is in Independent mode, we might need Indep policies.
        if l_mode == 0 or r_mode == 0:
            obs_i = env_shadow_i.task.get_observation(env_shadow_i.physics)
            img_i = obs_i['images']['top'].copy()
            img_t = torch.from_numpy(img_i).permute(2, 0, 1).unsqueeze(0).float().cuda() / 255.0
            
            # Left Independent
            if l_mode == 0:
                img_l = apply_torch_rgb_mask(img_t[:, :, :, :160], arm='left')
                qpos_l = (torch.from_numpy(qpos_main[:7]).float().cuda().unsqueeze(0) - torch.from_numpy(stats_l['qpos_mean']).float().cuda()) / torch.from_numpy(stats_l['qpos_std']).float().cuda()
                with torch.no_grad():
                    all_actions_l_out = policy_l(qpos_l, img_l.unsqueeze(1))[0].cpu().numpy()
                all_time_actions_l[t, t:t+chunk_size] = post_process_l(all_actions_l_out)

            # Right Independent
            if r_mode == 0:
                img_r = apply_torch_rgb_mask(img_t[:, :, :, 160:], arm='right')
                qpos_r = (torch.from_numpy(qpos_main[7:]).float().cuda().unsqueeze(0) - torch.from_numpy(stats_r['qpos_mean']).float().cuda()) / torch.from_numpy(stats_r['qpos_std']).float().cuda()
                with torch.no_grad():
                    all_actions_r_out = policy_r(qpos_r, img_r.unsqueeze(1))[0].cpu().numpy()
                all_time_actions_r[t, t:t+chunk_size] = post_process_r(all_actions_r_out)

        # CASE 2: Cooperative Policy
        # If either arm is in Cooperative mode, we need the Coop policy.
        if l_mode == 1 or r_mode == 1:
            obs_c = env_shadow_c.task.get_observation(env_shadow_c.physics)
            img_c = obs_c['images']['top'].copy()
            img_t = torch.from_numpy(img_c).permute(2, 0, 1).unsqueeze(0).float().cuda() / 255.0
            qpos_c = (torch.from_numpy(qpos_main).float().cuda().unsqueeze(0) - torch.from_numpy(stats_c['qpos_mean']).float().cuda()) / torch.from_numpy(stats_c['qpos_std']).float().cuda()
            
            with torch.no_grad():
                all_actions_c_out = policy_c(qpos_c, img_t.unsqueeze(1))[0].cpu().numpy()
            all_time_actions_c[t, t:t+chunk_size] = post_process_c(all_actions_c_out)

        # --- Temporal Ensembling & Selection ---
        start_idx = max(0, t - chunk_size + 1)
        curr_chunk_size = t + 1 - start_idx
        w = exp_weights[:curr_chunk_size][::-1]

        # Left Arm Action Selection
        if l_mode == 0: # Independent
            actions_step = all_time_actions_l[start_idx:t+1, t]
            mask = np.any(actions_step != 0, axis=1)
            w_masked = w * mask
            if w_masked.sum() > 0:
                act_l = np.sum(actions_step * (w_masked / w_masked.sum())[:, None], axis=0)
            else:
                act_l = all_time_actions_l[t, t]
        else: # Cooperative (use left part)
            actions_step = all_time_actions_c[start_idx:t+1, t, :7]
            mask = np.any(actions_step != 0, axis=1)
            w_masked = w * mask
            if w_masked.sum() > 0:
                act_l = np.sum(actions_step * (w_masked / w_masked.sum())[:, None], axis=0)
            else:
                act_l = all_time_actions_c[t, t, :7]

        # Right Arm Action Selection
        if r_mode == 0: # Independent
            actions_step = all_time_actions_r[start_idx:t+1, t]
            mask = np.any(actions_step != 0, axis=1)
            w_masked = w * mask
            if w_masked.sum() > 0:
                act_r = np.sum(actions_step * (w_masked / w_masked.sum())[:, None], axis=0)
            else:
                act_r = all_time_actions_r[t, t]
        else: # Cooperative (use right part)
            actions_step = all_time_actions_c[start_idx:t+1, t, 7:]
            mask = np.any(actions_step != 0, axis=1)
            w_masked = w * mask
            if w_masked.sum() > 0:
                act_r = np.sum(actions_step * (w_masked / w_masked.sum())[:, None], axis=0)
            else:
                act_r = all_time_actions_c[t, t, 7:]
        
        # Combine
        action = np.concatenate([act_l, act_r])
        
        ts = env_main.step(action)
        
        # 4. Success Detection
        sc = is_success(env_main.physics)
        if sc > 0:
            print(f"  [SUCCESS] {sc} cubes in goal area!")
            # If we want to wait for all, check sc == 10 or similar.
            # For now, let's just log and continue, or stop if it's the target.
        
        # 5. Rendering and Overlay
        if t % 100 == 0: print(f"  Step {t}/{max_steps} done. L_M:{l_mode} R_M:{r_mode}")
        
        if args.onscreen_render and t % 5 == 0:
            main_img = ts.observation['images']['top'].copy()
            # Draw HUD
            MODES = ["INDEP", "COOP/HOLD"]
            l_info = f"L: {MODES[l_mode]}"
            r_info = f"R: {MODES[r_mode]}"
            cv2.putText(main_img, l_info, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(main_img, r_info, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            plt_main.set_data(main_img)
            plt_c.set_data(env_shadow_c.physics.render(height=240, width=320, camera_id='top'))
            plt_i.set_data(env_shadow_i.physics.render(height=240, width=320, camera_id='top'))
            fig.canvas.draw(); fig.canvas.flush_events(); plt.pause(0.001)

    print("Finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', type=str, required=True)
    parser.add_argument('--cls_ckpt', type=str, required=True)
    parser.add_argument('--ckpt_c', type=str, required=True)
    parser.add_argument('--ckpt_l', type=str, required=True)
    parser.add_argument('--ckpt_r', type=str, required=True)
    parser.add_argument('--chunk_size', type=int, default=100)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--dim_feedforward', type=int, default=3200)
    parser.add_argument('--kl_weight', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--episode_len', type=int, default=None)
    parser.add_argument('--color_sequence', type=str, default=None)
    parser.add_argument('--onscreen_render', action='store_true')
    main(parser.parse_args())
