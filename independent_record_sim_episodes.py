import time
import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
import h5py

from piper_constants import PUPPET_GRIPPER_POSITION_NORMALIZE_FN, SIM_TASK_CONFIGS,BELT_MOVE_SPEED
from piper_ee_sim_env import make_ee_sim_env
from piper_sim_env import make_sim_env, REDBOX_POSE, GREENBOX_POSE, BLUEBOX_POSE
from scripted_policy import PickAndTransferPolicy, InsertionPolicy,PickMovingCubePolicy,CoopPolicy

import IPython
e = IPython.embed

# 右腕の開始姿勢（7次元）
RIGHT_ARM_START_POSE = np.array([-2.2, 1.1, -0.5, -1.9, -2.1, 1.0, 0])
# 左腕の開始姿勢（7次元）
LEFT_ARM_START_POSE = np.array([2.2, 1.1, -0.5, 1.9, -2.1, -0.8, 0])


def main(args):
    """
    Generate demonstration data in simulation.
    First rollout the policy (defined in ee space) in ee_sim_env. Obtain the joint trajectory.
    Replace the gripper joint positions with the commanded joint position.
    Replay this joint trajectory (as action sequence) in sim_env, and record all observations.
    Save this episode of data, and continue to next episode of data collection.
    """

    task_name = args['task_name']
    dataset_dir = args['dataset_dir']
    num_episodes = args['num_episodes']
    onscreen_render = args['onscreen_render']
    arm = args['arm']
    inject_noise = False
    render_cam_name = 'top'

    if arm == 'both':
        dataset_dir_left = dataset_dir + '_left'
        dataset_dir_right = dataset_dir + '_right'
        if not os.path.isdir(dataset_dir_left):
            os.makedirs(dataset_dir_left, exist_ok=True)
        if not os.path.isdir(dataset_dir_right):
            os.makedirs(dataset_dir_right, exist_ok=True)
    elif not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir, exist_ok=True)

    episode_len = SIM_TASK_CONFIGS[task_name]['episode_len']
    camera_names = SIM_TASK_CONFIGS[task_name]['camera_names']
    if task_name == 'sim_transfer_cube_scripted':
        policy_cls = PickAndTransferPolicy
    elif task_name == 'sim_insertion_scripted':
        policy_cls = InsertionPolicy
    elif task_name == 'sim_moving_cube_scripted':
        policy_cls = PickMovingCubePolicy
    elif task_name == 'sim_coop_scripted':
        policy_cls = CoopPolicy
    elif task_name == 'sim_independent_scripted':
        policy_cls = PickMovingCubePolicy
    else:
        raise NotImplementedError

    success = []
    for episode_idx in range(num_episodes):
        print(f'{episode_idx=}')
        print('Rollout out EE space scripted policy')
        # setup the environment
        env = make_ee_sim_env(task_name)
        ts = env.reset()
        episode = [ts]
        policy = policy_cls(inject_noise)
        # setup plotting
        all_actions = [] # actionを記録するための空リスト
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(ts.observation['images'][render_cam_name])
            plt.ion()

        for step in range(episode_len):
            action = policy(ts)
            #print(f"Step {step:01d} | EE Command: {action}")
            all_actions.append(action)
            ts = env.step(action)
            episode.append(ts)
            if onscreen_render:
                plt_img.set_data(ts.observation['images'][render_cam_name])
                plt.pause(0.002)
        plt.close()

        episode_return = np.sum([ts.reward for ts in episode[1:]])
        episode_max_reward = np.max([ts.reward for ts in episode[1:]])
        if episode_max_reward == env.task.max_reward:
            print(f"{episode_idx=} Successful, {episode_return=}")
        else:
            print(f"{episode_idx=} Failed")

        joint_traj = [ts.observation['qpos'] for ts in episode]

        # replace gripper pose with gripper control (片腕のみ)
        gripper_ctrl_traj = [ts.observation['gripper_ctrl'] for ts in episode]
        for joint, ctrl in zip(joint_traj, gripper_ctrl_traj):
            if arm == 'left':
                left_ctrl = PUPPET_GRIPPER_POSITION_NORMALIZE_FN(ctrl[0])
                joint[6] = left_ctrl
            elif arm == 'right':
                right_ctrl = PUPPET_GRIPPER_POSITION_NORMALIZE_FN(ctrl[0])
                joint[13] = right_ctrl
            elif arm == 'both':
                left_ctrl = PUPPET_GRIPPER_POSITION_NORMALIZE_FN(ctrl[0])
                joint[6] = left_ctrl
                right_ctrl = PUPPET_GRIPPER_POSITION_NORMALIZE_FN(ctrl[0])
                joint[13] = right_ctrl

        subtask_info = episode[0].observation['env_state'].copy() # box pose at step 0
        

        
        # clear unused variables
        del env
        del episode
        del policy

        #actions_array = np.array(all_actions)
        #np.savetxt("joint_traj.csv", joint_traj, delimiter=",", fmt="%.5f")


        # setup the environment
        print('Replaying joint commands')
        env = make_sim_env(task_name)
        REDBOX_POSE[0] = subtask_info[0:7].copy()      # red box
        GREENBOX_POSE[0] = subtask_info[7:14].copy()   # green box
        BLUEBOX_POSE[0] = subtask_info[14:21].copy()   # blue box
        ts = env.reset()

        all_actions = [] # actionを記録するための空リスト
        episode_replay = [ts]
        #start_arm_pose_np = np.array(START_ARM_POSE)
        # setup plotting
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(ts.observation['images'][render_cam_name])
            plt.ion()
        for t in range(len(joint_traj)): # note: this will increase episode length by 1
            joint_traj_np = np.array(joint_traj)
            action = joint_traj_np[t].copy()

            all_actions.append(action)
            ts = env.step(action)
            episode_replay.append(ts)
            if onscreen_render:
                plt_img.set_data(ts.observation['images'][render_cam_name])
                plt.pause(0.02)

        episode_return = np.sum([ts.reward for ts in episode_replay[1:]])
        episode_max_reward = np.max([ts.reward for ts in episode_replay[1:]])
        if episode_max_reward == env.task.max_reward:
            success.append(1)
            print(f"{episode_idx=} Successful, {episode_return=}")
        else:
            success.append(0)
            print(f"{episode_idx=} Failed")

        plt.close()
        joint_traj1 = [ts.observation['qpos'] for ts in episode_replay]
        #actions_array = np.array(all_actions)
        #np.savetxt("joint_traj1.csv", joint_traj1, delimiter=",", fmt="%.5f")

        """
        For each timestep:
        observations
        - images
            - each_cam_name     (480, 640, 3) 'uint8'
        - qpos                  (14,)         'float64'
        - qvel                  (14,)         'float64'

        action                  (14,)         'float64'
        """

        data_dict = {
            '/observations/qpos': [],
            '/observations/qvel': [],
            '/action': [],
        }
        for cam_name in camera_names:
            data_dict[f'/observations/images/{cam_name}'] = []

        # Prepare specific dictionaries based on 'arm'
        if arm == 'both':
            data_dict_left = {
                '/observations/qpos': [],
                '/observations/qvel': [],
                '/action': [],
            }
            data_dict_right = {
                '/observations/qpos': [],
                '/observations/qvel': [],
                '/action': [],
            }
            for cam_name in camera_names:
                data_dict_left[f'/observations/images/{cam_name}'] = []
                data_dict_right[f'/observations/images/{cam_name}'] = []
        else:
             # data_dict is already initialized above for single arm case
             pass

        # because the replaying, there will be eps_len + 1 actions and eps_len + 2 timesteps
        # truncate here to be consistent
        joint_traj = joint_traj[:-1]
        episode_replay = episode_replay[:-1]

        # len(joint_traj) i.e. actions: max_timesteps
        # len(episode_replay) i.e. time steps: max_timesteps + 1
        max_timesteps = len(joint_traj)
        while joint_traj:
            action = joint_traj.pop(0)
            ts = episode_replay.pop(0)
            
            if arm == 'left':
                data_dict['/observations/qpos'].append(ts.observation['qpos'][:7])
                data_dict['/observations/qvel'].append(ts.observation['qvel'][:7])
                data_dict['/action'].append(action[:7])
                for cam_name in camera_names:
                    # 左半分のみをトリミング (幅640の左半分320ピクセル)
                    img_left_half = ts.observation['images'][cam_name][:, :320, :]
                    data_dict[f'/observations/images/{cam_name}'].append(img_left_half)
            elif arm == 'right':
                data_dict['/observations/qpos'].append(ts.observation['qpos'][7:14])
                data_dict['/observations/qvel'].append(ts.observation['qvel'][7:14])
                data_dict['/action'].append(action[7:14])
                for cam_name in camera_names:
                    # 右半分のみをトリミング (幅640の右半分320ピクセル)
                    img_right_half = ts.observation['images'][cam_name][:, 320:, :]
                    data_dict[f'/observations/images/{cam_name}'].append(img_right_half)
            elif arm == 'both':
                # Left
                data_dict_left['/observations/qpos'].append(ts.observation['qpos'][:7])
                data_dict_left['/observations/qvel'].append(ts.observation['qvel'][:7])
                data_dict_left['/action'].append(action[:7])
                # Right
                data_dict_right['/observations/qpos'].append(ts.observation['qpos'][7:14])
                data_dict_right['/observations/qvel'].append(ts.observation['qvel'][7:14])
                data_dict_right['/action'].append(action[7:14])
                
                for cam_name in camera_names:
                    # Left Image
                    img_left_half = ts.observation['images'][cam_name][:, :320, :]
                    data_dict_left[f'/observations/images/{cam_name}'].append(img_left_half)
                    # Right Image
                    img_right_half = ts.observation['images'][cam_name][:, 320:, :]
                    data_dict_right[f'/observations/images/{cam_name}'].append(img_right_half)

        # HDF5 Saving
        t0 = time.time()
        
        def save_hdf5(dataset_path, data_container):
            with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
                root.attrs['sim'] = True
                obs = root.create_group('observations')
                image = obs.create_group('images')
                for cam_name in camera_names:
                    _ = image.create_dataset(cam_name, (max_timesteps, 480, 320, 3), dtype='uint8',
                                            chunks=(1, 480, 320, 3), )
                qpos = obs.create_dataset('qpos', (max_timesteps, 7))
                qvel = obs.create_dataset('qvel', (max_timesteps, 7))
                action = root.create_dataset('action', (max_timesteps, 7))

                for name, array in data_container.items():
                    root[name][...] = array

        if arm == 'both':
            save_hdf5(os.path.join(dataset_dir_left, f'episode_{episode_idx}'), data_dict_left)
            save_hdf5(os.path.join(dataset_dir_right, f'episode_{episode_idx}'), data_dict_right)
            print(f'Saved to {dataset_dir_left} and {dataset_dir_right}')
        else:
            save_hdf5(os.path.join(dataset_dir, f'episode_{episode_idx}'), data_dict)
            print(f'Saved to {dataset_dir}')
            
        print(f'Saving: {time.time() - t0:.1f} secs\n')

    print(f'Saved to {dataset_dir}')
    print(f'Success: {np.sum(success)} / {len(success)}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--dataset_dir', action='store', type=str, help='dataset saving dir', required=True)
    parser.add_argument('--num_episodes', action='store', type=int, help='num_episodes', required=False)
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--arm', action='store', type=str, help='arm', default='both', choices=['left', 'right', 'both'])
    
    main(vars(parser.parse_args()))

