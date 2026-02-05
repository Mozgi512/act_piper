import sys
import os
cwd = os.getcwd()
sys.path.append(os.path.join(cwd, 'detr'))

import torch
import numpy as np
import pickle
import argparse
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from einops import rearrange
import csv

from piper_constants import DT
from piper_constants import PUPPET_GRIPPER_JOINT_OPEN
from utils import load_data # data functions
from utils import sample_redbox_pose, sample_insertion_pose ,sample_greenbox_pose,sample_bluebox_pose, apply_rgb_mask_to_strip # robot functions
from utils import compute_dict_mean, set_seed, detach_dict # helper functions
from policy import ACTPolicy, CNNMLPPolicy
from visualize_episodes import save_videos

from piper_sim_env import REDBOX_POSE, GREENBOX_POSE, BLUEBOX_POSE, MANYCUBES_POSES, MANYCUBES_COLORS

import IPython
e = IPython.embed

# 右腕の開始姿勢（7次元）
RIGHT_ARM_START_POSE = np.array([-2.2, 1.1, -0.5, -1.9, -2.1, 1.0, 0])
# 左腕の開始姿勢（7次元）
LEFT_ARM_START_POSE = np.array([2.2, 1.1, -0.5, 1.9, -2.1, -1.0, 0])


def main(args):
    set_seed(1)
    
    # Set color sequence if provided
    if args['color_sequence']:
        MANYCUBES_COLORS[0] = args['color_sequence']
        print(f"Setting ManyCubes Color Sequence: {MANYCUBES_COLORS[0]}")

    # command line parameters
    is_eval = args['eval']
    ckpt_dir = args['ckpt_dir']
    policy_class = args['policy_class']
    onscreen_render = args['onscreen_render']
    task_name = args['task_name']
    batch_size_train = args['batch_size']
    batch_size_val = args['batch_size']
    num_epochs = args['num_epochs']
    arm = args['arm']

    # get task parameters
    is_sim = task_name[:4] == 'sim_'
    if is_sim:
        from piper_constants import SIM_TASK_CONFIGS
        task_config = SIM_TASK_CONFIGS[task_name]
    else:
        from aloha_scripts.constants import TASK_CONFIGS
        task_config = TASK_CONFIGS[task_name]
    if args['dataset_dir']:
        dataset_dir = args['dataset_dir']
    else:
        dataset_dir = task_config['dataset_dir']
    # 既存のディレクトリがない場合、armに応じたサフィックスを追加してチェック
    if not os.path.exists(dataset_dir):
        if os.path.exists(dataset_dir + f'_{arm}'):
             dataset_dir = dataset_dir + f'_{arm}'
        elif os.path.exists(dataset_dir + f'_left') and arm == 'left':
             dataset_dir = dataset_dir + '_left'
        elif os.path.exists(dataset_dir + f'_right') and arm == 'right':
             dataset_dir = dataset_dir + '_right'

    if args['num_episodes']:
        num_episodes = args['num_episodes']
    else:
        num_episodes = task_config['num_episodes']

    
    if args['episode_len']:
        episode_len = args['episode_len']
    else:
        episode_len = task_config['episode_len']
    camera_names = task_config['camera_names']

    # fixed parameters
    state_dim = 7  # 左腕のみ（7次元）
    lr_backbone = 1e-5
    backbone = 'resnet18'
    if policy_class == 'ACT':
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {
            'lr': args['lr'],
            'num_queries': args['chunk_size'],
            'kl_weight': args['kl_weight'],
            'hidden_dim': args['hidden_dim'],
            'dim_feedforward': args['dim_feedforward'],
            'lr_backbone': lr_backbone,
            'backbone': backbone,
            'enc_layers': enc_layers,
            'dec_layers': dec_layers,
            'nheads': nheads,
            'camera_names': camera_names,
            'state_dim': state_dim,  # この行を追加
            'arm': arm,
        }
    elif policy_class == 'CNNMLP':
        policy_config = {
            'lr': args['lr'],
            'camera_names': camera_names,
            'state_dim': state_dim,  # この行も追加
            'arm': arm,
        }
    else:
        raise NotImplementedError

    config = {
        'num_epochs': num_epochs,
        'ckpt_dir': ckpt_dir,
        'episode_len': episode_len,
        'state_dim': state_dim,
        'lr': args['lr'],
        'policy_class': policy_class,
        'onscreen_render': onscreen_render,
        'policy_config': policy_config,
        'task_name': task_name,
        'seed': args['seed'],
        'temporal_agg': args['temporal_agg'],
        'camera_names': camera_names,
        'real_robot': not is_sim,
        'real_robot': not is_sim,
        'arm': arm,
        'num_rollouts': args['num_rollouts'],
        'load_ckpt': args['load_ckpt']
    }

    if is_eval:
        ckpt_names = []
        if args['eval_epoch']:
            ckpt_names = [f'policy_epoch_{args["eval_epoch"]}_seed_{args["seed"]}.ckpt']
        elif args['eval_interval']:
            import re
            pattern = re.compile(r'policy_epoch_(\d+)_seed_\d+.ckpt')
            all_files = os.listdir(ckpt_dir)
            epoch_ckpts = []
            for filename in all_files:
                match = pattern.match(filename)
                if match:
                    epoch = int(match.group(1))
                    if epoch % args['eval_interval'] == 0:
                        epoch_ckpts.append((epoch, filename))
            
            epoch_ckpts.sort(key=lambda x: x[0])
            ckpt_names = [x[1] for x in epoch_ckpts]

        if 'policy_best.ckpt' not in ckpt_names and not args['eval_epoch']:
            ckpt_names.append('policy_best.ckpt')

        results = []
        for ckpt_name in ckpt_names:
            success_rate, avg_return = eval_bc(config, ckpt_name, save_episode=True)
            results.append([ckpt_name, success_rate, avg_return])

        for ckpt_name, success_rate, avg_return in results:
            print(f'{ckpt_name}: {success_rate=} {avg_return=}')
        
        # Save results to CSV
        csv_path = os.path.join(ckpt_dir, 'evaluation_results.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Checkpoint', 'Success Rate', 'Average Return'])
            writer.writerows(results)
        print(f'Saved evaluation results to {csv_path}')

        print()
        exit()

    train_dataloader, val_dataloader, stats, _ = load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val, args['num_workers'], args['prefetch_factor'], args['persistent_workers'], args['use_cache'])

    # save dataset stats
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)

    best_ckpt_info = train_bc(train_dataloader, val_dataloader, config)
    best_epoch, min_val_loss, best_state_dict = best_ckpt_info

    # save best checkpoint
    ckpt_path = os.path.join(ckpt_dir, f'policy_best.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Best ckpt, val loss {min_val_loss:.6f} @ epoch{best_epoch}')


def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    elif policy_class == 'CNNMLP':
        policy = CNNMLPPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy

def make_optimizer(policy_class, policy):
    if policy_class == 'ACT':
        optimizer = policy.configure_optimizers()
    elif policy_class == 'CNNMLP':
        optimizer = policy.configure_optimizers()
    else:
        raise NotImplementedError
    return optimizer


def apply_torch_rgb_mask(images, strip_width=40, arm='right'):
    """
    Applies RGB mask to a strip of the images (Batch, Cam, C, H, W) or (C, H, W).
    Preserves R, G, B colors; blacks out everything else in the strip.
    arm='right' -> Mask Left Strip (for Right Arm view).
    arm='left' -> Mask Right Strip (for Left Arm view).
    """
    # Handle single image or batch
    is_single = len(images.shape) == 3
    if is_single:
        images = images.unsqueeze(0)
    
    # Handle (B, Cam, C, H, W) -> flatten to (B*Cam, C, H, W)
    orig_shape = images.shape
    if len(orig_shape) == 5:
        b, n_cam, c, h, w = orig_shape
        images = images.view(b * n_cam, c, h, w)
        
    B, C, H, W = images.shape
    
    # Define Strip indices
    if arm == 'left':
        # Mask Right Strip (Overlap with Right Arm)
        strip_start = W - strip_width
        strip_end = W
    else:
        # Mask Left Strip (Overlap with Left Arm)
        strip_start = 0
        strip_end = strip_width
        
    strip = images[:, :, :, strip_start:strip_end]
    
    # Thresholds (matching utils.py cv2 logic: 100/255=0.392, 150/255=0.588)
    t_100 = 100.0 / 255.0
    t_150 = 150.0 / 255.0
    
    # Sim Env render gives RGB
    r = strip[:, 0, :, :]
    g = strip[:, 1, :, :]
    b = strip[:, 2, :, :]
    
    # Red: R > 100, G < 100, B < 100
    mask_r = (r > t_100) & (g < t_100) & (b < t_100)
    # Green: G > 100, R < 100, B < 100
    mask_g = (g > t_100) & (r < t_100) & (b < t_100)
    # Blue: B > 150, R < 100, G < 100
    mask_b = (b > t_150) & (r < t_100) & (g < t_100)
    
    combined_mask = mask_r | mask_g | mask_b # [B, H, W_strip]
    combined_mask = combined_mask.unsqueeze(1).repeat(1, C, 1, 1).float()
    
    masked_strip = strip * combined_mask
    
    outputs = images.clone()
    outputs[:, :, :, strip_start:strip_end] = masked_strip
    
    if len(orig_shape) == 5:
        outputs = outputs.view(orig_shape)
    elif is_single:
        outputs = outputs.squeeze(0)
        
    return outputs

def get_image(ts, camera_names, arm, device='cuda'):  
    curr_images = []
    for cam_name in camera_names:
        # ts.observation is numpy (H, W, C)
        curr_image_np = ts.observation['images'][cam_name].copy()
        h, w, c = curr_image_np.shape
        
        # Convert to Tensor (B, C, H, W)
        curr_image_t = torch.from_numpy(curr_image_np).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
        
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

def eval_bc(config, ckpt_name, save_episode=True):
    set_seed(1000)
    ckpt_dir = config['ckpt_dir']
    state_dim = config['state_dim']
    real_robot = config['real_robot']
    policy_class = config['policy_class']
    onscreen_render = config['onscreen_render']
    policy_config = config['policy_config']
    camera_names = config['camera_names']
    max_timesteps = config['episode_len']
    task_name = config['task_name']
    temporal_agg = config['temporal_agg']
    arm = config['arm']
    onscreen_cam = 'top'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load policy and stats
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = make_policy(policy_class, policy_config)
    loading_status = policy.load_state_dict(torch.load(ckpt_path, map_location=device))
    print(loading_status)
    policy.to(device)
    policy.eval()
    print(f'Loaded: {ckpt_path}')
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']

    # load environment
    if real_robot:
        from aloha_scripts.robot_utils import move_grippers # requires aloha
        from aloha_scripts.real_env import make_real_env # requires aloha
        env = make_real_env(init_node=True)
        env_max_reward = 0
    else:
        from piper_sim_env import make_sim_env
        env = make_sim_env(task_name)
        env_max_reward = env.task.max_reward

    query_frequency = policy_config['num_queries']
    if temporal_agg:
        query_frequency = 1
        num_queries = policy_config['num_queries']

    max_timesteps = int(max_timesteps * 1) # may increase for real-world tasks

    num_rollouts = config.get('num_rollouts', 50)
    episode_returns = []
    highest_rewards = []
    for rollout_id in range(num_rollouts):
        np.random.seed(rollout_id) # Force deterministic seed to match training data generation
        rollout_id += 0
        ### set task
        if 'sim_transfer_cube' in task_name:
            BOX_POSE[0] = sample_box_pose() # used in sim reset
        elif 'sim_insertion' in task_name:
            BOX_POSE[0] = np.concatenate(sample_insertion_pose()) # used in sim reset
        elif 'sim_moving_cube' in task_name:
            REDBOX_POSE[0] = sample_redbox_pose()      # red box
            GREENBOX_POSE[0] = sample_greenbox_pose()   # green box
            BLUEBOX_POSE[0] = sample_bluebox_pose()   # blue box
        elif 'sim_coop' in task_name:
            REDBOX_POSE[0] = sample_redbox_pose()      # red box
            GREENBOX_POSE[0] = sample_greenbox_pose()   # green box
            BLUEBOX_POSE[0] = sample_bluebox_pose()   # blue box
            
            # Override for Phase 2: Shift Red Box Left by 5cm (-0.05 X)
            if 'phase2' in task_name:
                poses = {}
                # Green (8) at goal
                poses[8] = np.array([0, 0.1, 0.025, 1, 0, 0, 0])
                # Blue (9) at goal
                poses[9] = np.array([0.08, 0.1, 0.025, 1, 0, 0, 0])
                # Red (7) at original (sample) + 0.2 offset + 0.02 shift = +0.22
                poses[7] = sample_redbox_pose() + np.array([0.22, 0, 0, 0, 0, 0, 0])
                MANYCUBES_POSES[0] = poses
                
            if 'phase2' in task_name:
                print('Warning: sim_coop_phase2 evaluation is not fully supported (reset to t=0).')
        elif 'sim_independent' in task_name:
            REDBOX_POSE[0] = sample_redbox_pose()      # red box
            GREENBOX_POSE[0] = sample_greenbox_pose()   # green box
            BLUEBOX_POSE[0] = sample_bluebox_pose()   # blue box
        ts = env.reset()
        
        # Override for Phase 2: Set robot to Handover Pose (t=280 of Phase 1)
        if 'phase2' in task_name and 'scripted' in task_name:
             # qpos extracted from get_phase2_start_pose.py (t=260)
             PHASE2_START_QPOS = np.array([ 0.9178,  1.852 , -1.7805,  2.1262, -0.97  , -1.8465,  1.    , 
                                           -0.9023,  1.9241, -1.4575, -1.8638, -0.7894,  1.4111,  0.5081])
             
             # Helper
             CLOSE = 0.005
             OPEN = 0.035
             unnorm = lambda x: x * (OPEN - CLOSE) + CLOSE
             
             new_qpos = np.zeros(16)
             # Left Arm
             new_qpos[0:6] = PHASE2_START_QPOS[0:6]
             # Left Gripper
             l_grip_val = unnorm(PHASE2_START_QPOS[6])
             new_qpos[6] = l_grip_val
             new_qpos[7] = -l_grip_val # Mirror
             
             # Right Arm
             new_qpos[8:14] = PHASE2_START_QPOS[7:13]
             # Right Gripper
             r_grip_val = unnorm(PHASE2_START_QPOS[13])
             new_qpos[14] = r_grip_val
             new_qpos[15] = -r_grip_val # Mirror
             
             env.physics.data.qpos[:16] = new_qpos
             
             # Ensure simulation state is consistent
             env.physics.forward() 
             
             # Reconstruct TS
             ts_obs = env.task.get_observation(env.physics)
             import dm_env
             ts = dm_env.TimeStep(
                step_type=ts.step_type,
                reward=ts.reward,
                discount=ts.discount,
                observation=ts_obs
             )


        ### onscreen render
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(env._physics.render(height=240, width=320, camera_id=onscreen_cam))
            plt.ion()

        ### evaluation loop
        if temporal_agg:
            all_time_actions = torch.zeros([max_timesteps, max_timesteps+num_queries, state_dim]).to(device)

        qpos_history = torch.zeros((1, max_timesteps, state_dim)).to(device)
        image_list = [] # for visualization
        qpos_list = []
        target_qpos_list = []
        rewards = []
        with torch.inference_mode():
            actions = []
            for t in range(max_timesteps):
                ### update onscreen render and wait for DT
                if onscreen_render:
                    image = env._physics.render(height=240, width=320, camera_id=onscreen_cam)
                    plt_img.set_data(image)
                    plt.pause(DT)

                ### process previous timestep to get qpos and image_list
                obs = ts.observation
                if 'images' in obs:
                    image_list.append(obs['images'])
                else:
                    image_list.append({'main': obs['image']})
                # 指定されたアームの情報を取得
                if arm == 'left':
                    qpos_numpy = np.array(obs['qpos'][:7])
                else:
                    qpos_numpy = np.array(obs['qpos'][7:14])
                qpos = pre_process(qpos_numpy)
                qpos = torch.from_numpy(qpos).float().to(device).unsqueeze(0)
                qpos_history[:, t] = qpos
                curr_image = get_image(ts, camera_names, arm, device=device)

                ### query policy
                if config['policy_class'] == "ACT":
                    if t % query_frequency == 0:
                        all_actions = policy(qpos, curr_image)
                    if temporal_agg:
                        all_time_actions[[t], t:t+num_queries] = all_actions
                        actions_for_curr_step = all_time_actions[:, t]
                        actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                        actions_for_curr_step = actions_for_curr_step[actions_populated]
                        k = 0.01
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = torch.from_numpy(exp_weights).to(device).unsqueeze(dim=1)
                        raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                    else:
                        raw_action = all_actions[:, t % query_frequency]
                elif config['policy_class'] == "CNNMLP":
                    raw_action = policy(qpos, curr_image)
                else:
                    raise NotImplementedError

                ### post-process actions
                raw_action = raw_action.squeeze(0).cpu().numpy()
                action = post_process(raw_action)
                
                # アクションと他方のアームの開始姿勢を結合して14次元にする
                if arm == 'left':
                    target_qpos = np.concatenate([action, RIGHT_ARM_START_POSE])
                else:
                    target_qpos = np.concatenate([LEFT_ARM_START_POSE, action])

                actions.append(action)

                ### step the environment
                ts = env.step(target_qpos)

                ### for visualization
                qpos_list.append(qpos_numpy)
                target_qpos_list.append(target_qpos)
                rewards.append(ts.reward)
            plt.close()
            np.savetxt("eval_actions.csv", actions, delimiter=",", fmt="%.5f")
        if real_robot:
            move_grippers([env.puppet_bot_left, env.puppet_bot_right], [PUPPET_GRIPPER_JOINT_OPEN] * 2, move_time=0.5)  # open
            pass

        rewards = np.array(rewards)
        episode_return = np.sum(rewards[rewards!=None])
        episode_returns.append(episode_return)
        episode_highest_reward = np.max(rewards)
        highest_rewards.append(episode_highest_reward)
        print(f'Rollout {rollout_id}\n{episode_return=}, {episode_highest_reward=}, {env_max_reward=}, Success: {episode_highest_reward==env_max_reward}')

        if save_episode:
            save_videos(image_list, DT, video_path=os.path.join(ckpt_dir, f'video{rollout_id}.mp4'))

    success_rate = np.mean(np.array(highest_rewards) == env_max_reward)
    avg_return = np.mean(episode_returns)
    summary_str = f'\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n'
    for r in range(env_max_reward+1):
        more_or_equal_r = (np.array(highest_rewards) >= r).sum()
        more_or_equal_r_rate = more_or_equal_r / num_rollouts
        summary_str += f'Reward >= {r}: {more_or_equal_r}/{num_rollouts} = {more_or_equal_r_rate*100}%\n'

    print(summary_str)

    # save success rate to txt
    result_file_name = 'result_' + ckpt_name.split('.')[0] + '.txt'
    with open(os.path.join(ckpt_dir, result_file_name), 'w') as f:
        f.write(summary_str)
        f.write(repr(episode_returns))
        f.write('\n\n')
        f.write(repr(highest_rewards))

    return success_rate, avg_return

def forward_pass(data, policy, arm, device='cuda', target_size=None):  
    image_data, qpos_data, action_data, is_pad = data
    image_data, qpos_data, action_data, is_pad = image_data.to(device), qpos_data.to(device), action_data.to(device), is_pad.to(device)
    
    # Normalize images (uint8 -> float32 [0, 1])
    image_data = image_data / 255.0
    
    # Apply RGB mask for consistency with inference
    image_data = apply_torch_rgb_mask(image_data, strip_width=40, arm=arm)
    
    if target_size is not None:
        # Resize images: [batch*cam, c, h, w] -> resize
        b, n_cam, c, h, w = image_data.shape
        image_data = image_data.view(b * n_cam, c, h, w)
        image_data = F.interpolate(image_data, size=target_size, mode='bilinear', align_corners=False)
        image_data = image_data.view(b, n_cam, c, target_size[0], target_size[1])
    
    # 14-dim splitting
    if qpos_data.shape[1] == 14:
        if arm == 'left':
            qpos_data = qpos_data[:, :7]
            action_data = action_data[:, :7]
        else:
            qpos_data = qpos_data[:, 7:14]
            action_data = action_data[:, 7:14]
            
    return policy(qpos_data, image_data, action_data, is_pad) # TODO remove None


def train_bc(train_dataloader, val_dataloader, config):
    num_epochs = config['num_epochs']
    ckpt_dir = config['ckpt_dir']
    seed = config['seed']
    policy_class = config['policy_class']
    policy_config = config['policy_config']
    arm = config.get('arm', 'left')

    set_seed(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    policy = make_policy(policy_class, policy_config)
    policy.to(device)

    if config['load_ckpt']:
        ckpt_path = config['load_ckpt']
        print(f'Loading checkpoint from {ckpt_path}...')
        state_dict = torch.load(ckpt_path)
        loading_status = policy.load_state_dict(state_dict)
        print(loading_status)
        print(f'Successfully loaded model weights from {ckpt_path}')

    if config.get('use_cuda_graph', False):
        print("Compiling model with torch.compile (mode='reduce-overhead')...")
        policy = torch.compile(policy, mode="reduce-overhead")

    optimizer = make_optimizer(policy_class, policy)
    scaler = torch.cuda.amp.GradScaler() # AMP scalar

    target_size = None
    if config.get('image_width') is not None and config.get('image_height') is not None:
        target_size = (config['image_height'], config['image_width'])

    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None
    for epoch in tqdm(range(num_epochs)):
        print(f'\nEpoch {epoch}')
        # validation
        if epoch % config.get('validation_interval', 100) == 0:
            with torch.inference_mode():
                policy.eval()
                epoch_dicts = []
                for batch_idx, data in enumerate(val_dataloader):
                    forward_dict = forward_pass(data, policy, arm, device=device, target_size=target_size)
                    epoch_dicts.append(forward_dict)
                epoch_summary = compute_dict_mean(epoch_dicts)
                validation_history.append(epoch_summary)

                epoch_val_loss = epoch_summary['loss']
                if epoch_val_loss < min_val_loss:
                    min_val_loss = epoch_val_loss
                    best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))
            print(f'Val loss:   {epoch_val_loss:.5f}')
            summary_string = ''
            for k, v in epoch_summary.items():
                summary_string += f'{k}: {v.item():.3f} '
            print(summary_string)

        # training
        policy.train()
        optimizer.zero_grad()

        for batch_idx, data in enumerate(train_dataloader):
            with torch.cuda.amp.autocast(): # AMP context
                forward_dict = forward_pass(data, policy, arm, device=device, target_size=target_size)
            # backward
            loss = forward_dict['loss']
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            train_history.append(detach_dict(forward_dict))
        epoch_summary = compute_dict_mean(train_history[(batch_idx+1)*epoch:(batch_idx+1)*(epoch+1)])
        epoch_train_loss = epoch_summary['loss']
        print(f'Train loss: {epoch_train_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)

        if epoch % 1000 == 0:
            ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{epoch}_seed_{seed}.ckpt')
            torch.save(policy.state_dict(), ckpt_path)
            plot_history(train_history, validation_history, epoch, ckpt_dir, seed)

    ckpt_path = os.path.join(ckpt_dir, f'policy_last.ckpt')
    torch.save(policy.state_dict(), ckpt_path)

    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{best_epoch}_seed_{seed}.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}')

    # save training curves
    plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed)

    return best_ckpt_info


def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    # save training curves
    for key in train_history[0]:
        plot_path = os.path.join(ckpt_dir, f'train_val_{key}_seed_{seed}.png')
        plt.figure()
        train_values = [summary[key].item() for summary in train_history]
        val_values = [summary[key].item() for summary in validation_history]
        plt.plot(np.linspace(0, num_epochs-1, len(train_history)), train_values, label='train')
        plt.plot(np.linspace(0, num_epochs-1, len(validation_history)), val_values, label='validation')
        # plt.ylim([-0.1, 1])
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
    print(f'Saved plots to {ckpt_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', action='store', type=str, help='dataset_dir', required=False)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=True)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', required=True)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=True)
    parser.add_argument('--lr', action='store', type=float, help='lr', required=True)
    parser.add_argument('--arm', action='store', type=str, help='arm', default='left', choices=['left', 'right'])

    parser.add_argument('--num_workers', action='store', type=int, help='num_workers', required=False, default=1)
    parser.add_argument('--prefetch_factor', action='store', type=int, help='prefetch_factor', required=False, default=2)
    parser.add_argument('--persistent_workers', action='store_true', help='persistent_workers', required=False)
    parser.add_argument('--use_cache', action='store_true', help='cache dataset in memory', required=False)
    parser.add_argument('--use_cuda_graph', action='store_true', help='use torch.compile with reduce-overhead', required=False)
    parser.add_argument('--validation_interval', action='store', type=int, help='validation interval', required=False, default=100)
    parser.add_argument('--image_width', action='store', type=int, help='image width', required=False)
    parser.add_argument('--image_height', action='store', type=int, help='image height', required=False)
    parser.add_argument('--eval_interval', action='store', type=int, help='eval interval', required=False)
    parser.add_argument('--num_rollouts', action='store', type=int, help='number of rollouts for eval', required=False, default=50)
    parser.add_argument('--num_episodes', action='store', type=int, help='number of episodes to use', required=False)
    parser.add_argument('--eval_epoch', action='store', type=int, help='specific epoch to eval', required=False)
    parser.add_argument('--load_ckpt', action='store', type=str, help='Checkpoint path to load weights from', default=None)
    parser.add_argument('--episode_len', action='store', type=int, help='Override task episode length', required=False)

    # for ACT
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', required=False)
    parser.add_argument('--temporal_agg', action='store_true')
    parser.add_argument('--color_sequence', action='store', type=str, help='Color sequence for many_cubes task (e.g. rgrg)', default=None)
    
    main(vars(parser.parse_args()))
