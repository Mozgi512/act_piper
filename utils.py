import numpy as np
import torch
import os
import h5py
from torch.utils.data import TensorDataset, DataLoader

import cv2

def apply_rgb_mask_to_strip(image, strip_width=40):
    """
    Applies a color mask to the leftmost `strip_width` pixels of the image.
    Preserves Red, Green, and Blue colors; blacks out everything else.
    Image is expected to be (H, W, 3) numpy array (uint8).
    """
    if image.shape[1] < strip_width:
        return image
        
    strip = image[:, :strip_width, :]
    
    lower_red = np.array([100, 0, 0], dtype=np.uint8)
    upper_red = np.array([255, 100, 100], dtype=np.uint8)
    
    lower_green = np.array([0, 100, 0], dtype=np.uint8)
    upper_green = np.array([100, 255, 100], dtype=np.uint8)
    
    # Floor blue max is ~102, so use 150 to be safe
    lower_blue = np.array([0, 0, 150], dtype=np.uint8)
    upper_blue = np.array([100, 100, 255], dtype=np.uint8)
    
    mask_r = cv2.inRange(strip, lower_red, upper_red)
    mask_g = cv2.inRange(strip, lower_green, upper_green)
    mask_b = cv2.inRange(strip, lower_blue, upper_blue)
    
    combined_mask = cv2.bitwise_or(mask_r, mask_g)
    combined_mask = cv2.bitwise_or(combined_mask, mask_b)
    
    masked_strip = cv2.bitwise_and(strip, strip, mask=combined_mask)
    
    image[:, :strip_width, :] = masked_strip
    return image

def apply_rgb_mask_to_right_strip(image, strip_width=40):
    """
    Applies a color mask to the rightmost `strip_width` pixels of the image.
    Preserves Red, Green, and Blue colors; blacks out everything else.
    Image is expected to be (H, W, 3) numpy array (uint8).
    """
    if image.shape[1] < strip_width:
        return image
        
    strip = image[:, -strip_width:, :]
    
    lower_red = np.array([100, 0, 0], dtype=np.uint8)
    upper_red = np.array([255, 100, 100], dtype=np.uint8)
    
    lower_green = np.array([0, 100, 0], dtype=np.uint8)
    upper_green = np.array([100, 255, 100], dtype=np.uint8)
    
    # Floor blue max is ~102, so use 150 to be safe
    lower_blue = np.array([0, 0, 150], dtype=np.uint8)
    upper_blue = np.array([100, 100, 255], dtype=np.uint8)
    
    mask_r = cv2.inRange(strip, lower_red, upper_red)
    mask_g = cv2.inRange(strip, lower_green, upper_green)
    mask_b = cv2.inRange(strip, lower_blue, upper_blue)
    
    combined_mask = cv2.bitwise_or(mask_r, mask_g)
    combined_mask = cv2.bitwise_or(combined_mask, mask_b)
    
    masked_strip = cv2.bitwise_and(strip, strip, mask=combined_mask)
    
    image[:, -strip_width:, :] = masked_strip
    return image

import IPython
e = IPython.embed

class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats, use_cache=False):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.is_sim = None
        self.use_cache = use_cache
        self.cache = {}
        self._files = {} # Lazy file cache
        self.max_cached_files = 50 # Avoid hitting uimit
        
        if self.use_cache:
            print(f"Pre-loading low-dim data for {len(episode_ids)} episodes into RAM...")
            from tqdm import tqdm
            for episode_id in tqdm(episode_ids):
                dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')
                with h5py.File(dataset_path, 'r') as root:
                    is_sim = root.attrs['sim']
                    qpos = root['/observations/qpos'][()]
                    qvel = root['/observations/qvel'][()]
                    action = root['/action'][()]
                    
                self.cache[episode_id] = {
                    'is_sim': is_sim,
                    'qpos': qpos,
                    'qvel': qvel,
                    'action': action,
                }
            print("Cache loading complete (Images will be read from disk).")
        
        self.__getitem__(0) # initialize self.is_sim

    def __len__(self):
        return len(self.episode_ids)

    def _get_file_handle(self, episode_id):
        if episode_id not in self._files:
            # simple LRU: if too many files, close random/first
            if len(self._files) >= self.max_cached_files:
                closed_id = next(iter(self._files))
                self._files[closed_id].close()
                del self._files[closed_id]
                
            dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')
            self._files[episode_id] = h5py.File(dataset_path, 'r', libver='latest', swmr=True)
            
        return self._files[episode_id]

    def __getitem__(self, index):
        sample_full_episode = False # hardcode

        episode_id = self.episode_ids[index]
        
        # 1. Get Low-Dim Data
        if self.use_cache and episode_id in self.cache:
            # Hit RAM cache
            data = self.cache[episode_id]
            is_sim = data['is_sim']
            qpos_all = data['qpos']
            action_all = data['action']
            
            original_action_shape = action_all.shape
            episode_len = original_action_shape[0]
            if sample_full_episode:
                start_ts = 0
            else:
                start_ts = np.random.choice(episode_len)
            
            qpos = qpos_all[start_ts]
            
            if is_sim:
                action = action_all[start_ts:]
                action_len = episode_len - start_ts
            else:
                action = action_all[max(0, start_ts - 1):]
                action_len = episode_len - max(0, start_ts - 1)
                
            # Need file for images anyway
            root = self._get_file_handle(episode_id)
            image_dict = dict()
            for cam_name in self.camera_names:
                image_dict[cam_name] = root[f'/observations/images/{cam_name}'][start_ts]

        else:
            # No RAM cache, read everything from file
            # Use cached file handle to avoid open() overhead
            root = self._get_file_handle(episode_id)
            
            is_sim = root.attrs['sim']
            original_action_shape = root['/action'].shape
            episode_len = original_action_shape[0]
            if sample_full_episode:
                start_ts = 0
            else:
                start_ts = np.random.choice(episode_len)
            
            qpos = root['/observations/qpos'][start_ts]
            qvel = root['/observations/qvel'][start_ts]
            
            image_dict = dict()
            for cam_name in self.camera_names:
                image_dict[cam_name] = root[f'/observations/images/{cam_name}'][start_ts]
            
            if is_sim:
                action = root['/action'][start_ts:]
                action_len = episode_len - start_ts
            else:
                action = root['/action'][max(0, start_ts - 1):]
                action_len = episode_len - max(0, start_ts - 1)

        self.is_sim = is_sim
        padded_action = np.zeros(original_action_shape, dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.zeros(episode_len)
        is_pad[action_len:] = 1

        if self.is_sim:
            # Check for masking requirement (Independent datasets)
            pass

        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)

        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # channel last
        image_data = torch.einsum('k h w c -> k c h w', image_data)

        # normalize image and change dtype to float
        # image_data = image_data / 255.0
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        return image_data, qpos_data, action_data, is_pad


def get_norm_stats(dataset_dir, num_episodes):
    all_qpos_data = []
    all_action_data = []
    for episode_idx in range(num_episodes):
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            qpos = root['/observations/qpos'][()]
            qvel = root['/observations/qvel'][()]
            action = root['/action'][()]
        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))
    all_qpos_data = torch.stack(all_qpos_data)
    all_action_data = torch.stack(all_action_data)
    all_action_data = all_action_data

    # normalize action data
    action_mean = all_action_data.mean(dim=[0, 1], keepdim=True)
    action_std = all_action_data.std(dim=[0, 1], keepdim=True)
    action_std = torch.clip(action_std, 1e-2, np.inf) # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=[0, 1], keepdim=True)
    qpos_std = all_qpos_data.std(dim=[0, 1], keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf) # clipping

    stats = {"action_mean": action_mean.numpy().squeeze(), "action_std": action_std.numpy().squeeze(),
             "qpos_mean": qpos_mean.numpy().squeeze(), "qpos_std": qpos_std.numpy().squeeze(),
             "example_qpos": qpos}

    return stats


def load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val, num_workers=1, prefetch_factor=2, persistent_workers=False, use_cache=False):
    print(f'\nData from: {dataset_dir}\n')
    # obtain train test split
    train_ratio = 0.8
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    # obtain normalization stats for qpos and action
    norm_stats = get_norm_stats(dataset_dir, num_episodes)

    # construct dataset and dataloader
    train_dataset = EpisodicDataset(train_indices, dataset_dir, camera_names, norm_stats, use_cache=use_cache)
    val_dataset = EpisodicDataset(val_indices, dataset_dir, camera_names, norm_stats, use_cache=use_cache)
    
    # Check if prefetch_factor is valid (requires num_workers > 0)
    if num_workers == 0:
        prefetch_factor = None 
        persistent_workers = False

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=num_workers, prefetch_factor=prefetch_factor, persistent_workers=persistent_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=num_workers, prefetch_factor=prefetch_factor, persistent_workers=persistent_workers)

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim


### env utils

def sample_redbox_pose():
    x_range = [-0.45, -0.30]
    y_range = [0.30, 0.4]
    z_range = [0.005, 0.005]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])


def sample_greenbox_pose():
    x_range = [-0.25,-0.10]
    y_range = [0.30, 0.4]
    z_range = [0.005, 0.005]

    ranges = np.vstack([x_range, y_range, z_range])
    stick_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    stick_quat = np.array([1, 0, 0, 0])
    return np.concatenate([stick_position, stick_quat])

def sample_bluebox_pose():
    x_range = [0.00, 0.15]
    y_range = [0.30, 0.4]
    z_range = [0.005, 0.005]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    return np.concatenate([socket_position, socket_quat])


def sample_redbox1_pose():
    x_range = [-0.30, -0.25]
    y_range = [0.30, 0.4]
    z_range = [0.005, 0.005]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

def sample_redbox2_pose():
    x_range = [-0.40, -0.35]
    y_range = [0.30, 0.4]
    z_range = [0.005, 0.005]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

def sample_greenbox1_pose():
    x_range = [-0.20,-0.05]
    y_range = [0.30, 0.4]
    z_range = [0.005, 0.005]

    ranges = np.vstack([x_range, y_range, z_range])
    stick_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    stick_quat = np.array([1, 0, 0, 0])
    return np.concatenate([stick_position, stick_quat])

def sample_bluebox1_pose():
    x_range = [0.00, 0.15]
    y_range = [0.30, 0.4]
    z_range = [0.005, 0.005]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    return np.concatenate([socket_position, socket_quat])
def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.05, 0.25]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.05, 0.25]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose

### helper functions

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach().cpu()
    return new_d

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def sample_cube_pose(x_range, y_range):
    z_range = [0.01, 0.01]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])
