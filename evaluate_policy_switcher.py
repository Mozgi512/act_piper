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
from utils import apply_rgb_mask_to_strip
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
        
        if arm == 'left':
            curr_image = curr_image[:, :w//2, :]
        else:
            # Shift based on width (adaptive)
            # Standard offset=40 for 640.
            offset = int(40 * (w / 640))
            
            start = w//2 - offset
            end = w - offset
            curr_image = curr_image[:, start:end, :]
            
            # Apply RGB mask
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

    env = make_sim_env(task_name)
    
    if onscreen_render:
        plt.ion()
        ax = plt.subplot()
        plt_img = ax.imshow(env._physics.render(height=480, width=640, camera_id='top'))
    
    # State tracking
    chunk_size = args.chunk_size
    
    # Initialize Temporal Aggregation
    temporal_agg = not args.no_temporal_agg
    num_queries = args.chunk_size
    # We need a large buffer, or rolling buffer. Since max_timesteps is dynamic (sum of subtasks), let's make it large enough.
    # Assumed max total steps = 4000
    MAX_BUFFER_STEPS = 5000
    float_nan = float('nan')
    all_time_actions_dual = torch.full([MAX_BUFFER_STEPS, MAX_BUFFER_STEPS+num_queries, 14], float_nan).cuda()
    all_time_actions_left = torch.full([MAX_BUFFER_STEPS, MAX_BUFFER_STEPS+num_queries, 7], float_nan).cuda()
    all_time_actions_right = torch.full([MAX_BUFFER_STEPS, MAX_BUFFER_STEPS+num_queries, 7], float_nan).cuda()

    episode_count = 0
    
    try:
        while episode_count < args.num_rollouts:
            # Reset Environment
            ts = env.reset()
            t = 0
            
            # Reset buffers
            all_time_actions_dual.fill_(float_nan)
            all_time_actions_left.fill_(float_nan)
            all_time_actions_right.fill_(float_nan)
            
            step_in_chunk = 0
            current_action_chunk_dual = None
            current_action_chunk_left = None
            current_action_chunk_right = None
            
            video_frames = []
            
            # Execution Queue
            current_queue = list(command_queue)
            current_subtask_end = -1
            current_mode = None # 'I' or 'C'
            
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
                            # ... (Same logic as policy_switcher) ...
                            if old_mode == MODE_COOP and current_mode == MODE_INDEPENDENT:
                                # Dual -> Indep
                                input_actions = all_time_actions_dual
                                dual_mask = ~torch.isnan(input_actions)
                                input_actions_safe = torch.nan_to_num(input_actions, nan=0.0)
                                denorm_actions = input_actions_safe * stats_dual_torch['action_std'] + stats_dual_torch['action_mean']
                                
                                # Left
                                denorm_left = denorm_actions[:, :, :7]
                                renorm_left = (denorm_left - stats_left_torch['action_mean']) / stats_left_torch['action_std']
                                val_inherit_left = renorm_left.clone()
                                val_inherit_left[~dual_mask[:, :, :7]] = float('nan')
                                all_time_actions_left.copy_(val_inherit_left)
                                
                                # Right
                                denorm_right = denorm_actions[:, :, 7:]
                                renorm_right = (denorm_right - stats_right_torch['action_mean']) / stats_right_torch['action_std']
                                val_inherit_right = renorm_right.clone()
                                val_inherit_right[~dual_mask[:, :, 7:]] = float('nan')
                                all_time_actions_right.copy_(val_inherit_right)
                                
                            elif old_mode == MODE_INDEPENDENT and current_mode == MODE_COOP:
                                # Indep -> Dual
                                input_left = all_time_actions_left
                                input_right = all_time_actions_right
                                left_mask = ~torch.isnan(input_left)
                                right_mask = ~torch.isnan(input_right)
                                combined_mask = torch.cat([left_mask, right_mask], dim=2)
                                
                                input_left_safe = torch.nan_to_num(input_left, nan=0.0)
                                input_right_safe = torch.nan_to_num(input_right, nan=0.0)
                                denorm_left = input_left_safe * stats_left_torch['action_std'] + stats_left_torch['action_mean']
                                denorm_right = input_right_safe * stats_right_torch['action_std'] + stats_right_torch['action_mean']
                                
                                denorm_dual = torch.cat([denorm_left, denorm_right], dim=2)
                                renorm_dual = (denorm_dual - stats_dual_torch['action_mean']) / stats_dual_torch['action_std']
                                
                                val_inherit_dual = renorm_dual.clone()
                                val_inherit_dual[~combined_mask] = float('nan')
                                all_time_actions_dual.copy_(val_inherit_dual)

                    else:
                        print(f"[Step {t}] All commands finished.")
                        break

                # -------------------------------
                # Render & Obs
                # -------------------------------
                if onscreen_render:
                    image = env._physics.render(height=480, width=640, camera_id='top')
                    plt_img.set_data(image)
                    plt.pause(DT)

                if args.save_video:
                     if not onscreen_render:
                          image = env._physics.render(height=480, width=640, camera_id='top')
                     video_frames.append(image) 
                
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

                        if current_mode == MODE_COOP:
                            qpos = pre_process_dual(qpos_numpy)
                            qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                            curr_image = get_image_dual(ts, camera_names)
                            
                            action_chunk = policy_dual(qpos, curr_image)
                            if temporal_agg:
                                all_time_actions_dual[[t], t:t+num_queries] = action_chunk
                            else:
                                current_action_chunk_dual = action_chunk.squeeze(0).cpu().numpy()
                        else:
                            # Independent
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
                        
                        target_qpos = post_process_dual(raw_action)
                        
                    else:
                        # Independent
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
                
                step_in_chunk += 1
                t += 1
                
                # Safety break
                if t >= MAX_BUFFER_STEPS:
                    print("Max steps reached. Terminating.")
                    break
            
            # End of Episode
            if args.save_video and len(video_frames) > 0:
                 video_path = f'eval_switch_ep{episode_count}.mp4'
                 h, w, _ = video_frames[0].shape
                 fps = 30
                 out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                 for frame in video_frames:
                     frame_bgr = frame[:, :, [2, 1, 0]]
                     out.write(frame_bgr)
                 out.release()
                 print(f"Saved video to {video_path}")
            
            episode_count += 1

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', action='store', type=str, default='sim_many_cubes')
    parser.add_argument('--ckpt_dual', action='store', type=str, required=True)
    parser.add_argument('--ckpt_left', action='store', type=str, required=True)
    parser.add_argument('--ckpt_right', action='store', type=str, required=True)
    
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
    parser.add_argument('--save_video', action='store_true', help='Save execution video')
    parser.add_argument('--num_rollouts', action='store', type=int, default=1, help='Number of evaluation episodes')
    
    args = parser.parse_args()
    main(args)
