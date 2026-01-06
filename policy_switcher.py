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

# Add detr to path
cwd = os.getcwd()
sys.path.append(os.path.join(cwd, 'detr'))

# Import existing modules
from piper_constants import DT
from piper_constants import PUPPET_GRIPPER_JOINT_OPEN
from utils import load_data
from utils import sample_redbox_pose, sample_insertion_pose, sample_greenbox_pose, sample_bluebox_pose
from utils import compute_dict_mean, set_seed, detach_dict
from policy import ACTPolicy, CNNMLPPolicy

# Import Sim Env
from piper_sim_env import REDBOX_POSE, GREENBOX_POSE, BLUEBOX_POSE
from piper_sim_env import make_sim_env

# Constants
MODE_INDEPENDENT = '1'
MODE_DUAL = '2'

def is_data():
    return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])

def get_key():
    if is_data():
        return sys.stdin.read(1)
    return None

def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    elif policy_class == 'CNNMLP':
        policy = CNNMLPPolicy(policy_config)
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
        curr_image = rearrange(ts.observation['images'][cam_name], 'h w c -> c h w')
        _, h, w = curr_image.shape
        if arm == 'left':
            curr_image = curr_image[:, :, :w//2]
        else:
            curr_image = curr_image[:, :, w//2:]
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

    if policy_class == 'CNNMLP':
         policy_config = {'lr': args.lr, 'lr_backbone': lr_backbone, 'backbone' : backbone, 'num_queries': 1,
                          'camera_names': camera_names, 'state_dim': state_dim}
         if override_arm:
            policy_config['arm'] = override_arm

    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    ckpt_path = os.path.join(ckpt_dir, 'policy_best.ckpt')
    policy = make_policy(policy_class, policy_config)
    policy.load_state_dict(torch.load(ckpt_path))
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
    
    is_sim = task_name[:4] == 'sim_'
    if not is_sim:
        print("Real robot support not implemented in this switcher yet.")
        return

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
        return env.reset()

    ts = reset_with_new_pose()
    
    old_settings = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())
    
    # Default to Independent logic or Dual logic depending on request?
    # User changed it to MODE_DUAL in Step 162. Keeping MODE_DUAL.
    current_mode = MODE_DUAL
    
    print("\n\nReady!")
    print("Press '1' for Independent Mode")
    print("Press '2' for Dual Mode")
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
        all_time_actions_dual = torch.zeros([max_timesteps, max_timesteps+num_queries, 14]).cuda()
        all_time_actions_left = torch.zeros([max_timesteps, max_timesteps+num_queries, 7]).cuda()
        all_time_actions_right = torch.zeros([max_timesteps, max_timesteps+num_queries, 7]).cuda()


    try:
        t = 0
        while True:
            key = get_key()
            if key == '1':
                if current_mode != MODE_INDEPENDENT:
                    current_mode = MODE_INDEPENDENT
                    step_in_chunk = 0 # Force replan on switch
                    print(f"[Step {t}] Switched to INDEPENDENT mode")
            elif key == '2':
                if current_mode != MODE_DUAL:
                    current_mode = MODE_DUAL
                    step_in_chunk = 0 # Force replan on switch
                    print(f"[Step {t}] Switched to DUAL arm mode")
            elif key == 'q':
                break
                
            # Render update matching imitate_episodes.py timing
            if onscreen_render:
                image = env._physics.render(height=480, width=640, camera_id='top')
                plt_img.set_data(image)
                plt.pause(DT) 
            
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

                    if current_mode == MODE_DUAL:
                         qpos = pre_process_dual(qpos_numpy)
                         qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                         curr_image = get_image_dual(ts, camera_names)
                         
                         if args.policy_class == 'ACT':
                             action_chunk = policy_dual(qpos, curr_image) # [1, chunk_size, 14]
                             if temporal_agg:
                                 all_time_actions_dual[[t], t:t+num_queries] = action_chunk
                             else:
                                 current_action_chunk_dual = action_chunk.squeeze(0).cpu().numpy()
                         else:
                             action = policy_dual(qpos, curr_image)
                             if temporal_agg:
                                 all_time_actions_dual[[t], t:t+num_queries] = action # CNNMLP outputs 1 step but shaped [1, 1, 14] maybe? No ACT outputs chunk.
                                 # CNNMLP usually outputs [1, 14], let's assume it supports chunking or we handle it. 
                                 # In original code: policy_config = {..., 'num_queries': 1, ...} for CNNMLP
                                 # So action is [1, 14]. 
                                 # Wait, existing code says: action = policy_dual(qpos, curr_image) -> current_action_chunk_dual = action.cpu().numpy() 
                                 # Then raw_action = current_action_chunk_dual[0]. 
                                 # If temporal_agg is used with CNNMLP it effectively averages 1 value? Usually temporal_agg is for ACT.
                                 # Let's support ACT mainly for temporal agg as per original script.
                                 pass 
                             else:
                                 current_action_chunk_dual = action.cpu().numpy() 

                    else:
                        # Left
                         qpos_left_numpy = qpos_numpy[:7]
                         qpos_left = pre_process_left(qpos_left_numpy)
                         qpos_left = torch.from_numpy(qpos_left).float().cuda().unsqueeze(0)
                         curr_image_left = get_image_independent(ts, camera_names, 'left')
                         
                         if args.policy_class == 'ACT':
                             action_chunk_l = policy_left(qpos_left, curr_image_left)
                             if temporal_agg:
                                 all_time_actions_left[[t], t:t+num_queries] = action_chunk_l
                             else:
                                 current_action_chunk_left = action_chunk_l.squeeze(0).cpu().numpy()
                         else:
                             action_l = policy_left(qpos_left, curr_image_left)
                             if not temporal_agg:
                                 current_action_chunk_left = action_l.cpu().numpy()
                        
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
                         else:
                             action_r = policy_right(qpos_right, curr_image_right)
                             if not temporal_agg:
                                 current_action_chunk_right = action_r.cpu().numpy()
                             
                
                # Execute current step of the plan
                if current_mode == MODE_DUAL:
                    if args.policy_class == 'ACT':
                        if temporal_agg:
                            actions_for_curr_step = all_time_actions_dual[:, t]
                            actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                            actions_for_curr_step = actions_for_curr_step[actions_populated]
                            k = 0.01
                            exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                            exp_weights = exp_weights / exp_weights.sum()
                            exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                            raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                            raw_action = raw_action.squeeze(0).cpu().numpy()
                        else:
                            raw_action = current_action_chunk_dual[step_in_chunk]
                    else:
                        # CNNMLP
                        if temporal_agg:
                             # Fallback or simple forward for CNNMLP (usually no agg needed or simple 1 step)
                             # Original script 'imitate_episodes' handles CNNMLP by just: raw_action = policy(qpos, curr_image)
                             if should_plan: # already computed action
                                 raw_action = action.squeeze(0).cpu().numpy()
                        else:
                            raw_action = current_action_chunk_dual[0]
                    
                    action = post_process_dual(raw_action)
                    target_qpos = action
                else:
                    # Independent Mode
                    if args.policy_class == 'ACT':
                        if temporal_agg:
                            # LEFT
                            actions_for_curr_step_l = all_time_actions_left[:, t]
                            actions_populated_l = torch.all(actions_for_curr_step_l != 0, axis=1)
                            actions_for_curr_step_l = actions_for_curr_step_l[actions_populated_l]
                            k = 0.01
                            exp_weights_l = np.exp(-k * np.arange(len(actions_for_curr_step_l)))
                            exp_weights_l = exp_weights_l / exp_weights_l.sum()
                            exp_weights_l = torch.from_numpy(exp_weights_l).cuda().unsqueeze(dim=1)
                            raw_action_l = (actions_for_curr_step_l * exp_weights_l).sum(dim=0, keepdim=True)
                            raw_action_l = raw_action_l.squeeze(0).cpu().numpy()

                            # RIGHT
                            actions_for_curr_step_r = all_time_actions_right[:, t]
                            actions_populated_r = torch.all(actions_for_curr_step_r != 0, axis=1)
                            actions_for_curr_step_r = actions_for_curr_step_r[actions_populated_r]
                            exp_weights_r = np.exp(-k * np.arange(len(actions_for_curr_step_r)))
                            exp_weights_r = exp_weights_r / exp_weights_r.sum()
                            exp_weights_r = torch.from_numpy(exp_weights_r).cuda().unsqueeze(dim=1)
                            raw_action_r = (actions_for_curr_step_r * exp_weights_r).sum(dim=0, keepdim=True)
                            raw_action_r = raw_action_r.squeeze(0).cpu().numpy()
                        else:
                            raw_action_l = current_action_chunk_left[step_in_chunk]
                            raw_action_r = current_action_chunk_right[step_in_chunk]
                    else:
                        # CNNMLP
                        if temporal_agg:
                            if should_plan:
                                raw_action_l = action_l.squeeze(0).cpu().numpy()
                                raw_action_r = action_r.squeeze(0).cpu().numpy()
                        else:
                            raw_action_l = current_action_chunk_left[0]
                            raw_action_r = current_action_chunk_right[0]
                        
                    action_left = post_process_left(raw_action_l)
                    action_right = post_process_right(raw_action_r)
                    target_qpos = np.concatenate([action_left, action_right])
            
            ts = env.step(target_qpos)
            
            step_in_chunk += 1
            t += 1
            
            # Reset logic matches imitate_episodes num_rollouts loop (conceptually)
            if t >= max_timesteps:
                print("Episode finished. Resetting...")
                ts = reset_with_new_pose()
                t = 0
                step_in_chunk = 0

    finally:
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
    parser.add_argument('--onscreen_render', action='store_true', default=True)
    
    args = parser.parse_args()
    main(args)
