import numpy as np
import os
import collections
import matplotlib.pyplot as plt
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base

from piper_constants import DT, XML_DIR, START_ARM_POSE,BELT_MOVE_SPEED
from piper_constants import PUPPET_GRIPPER_POSITION_CLOSE
from piper_constants import PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN
from piper_constants import MASTER_GRIPPER_POSITION_NORMALIZE_FN
from piper_constants import PUPPET_GRIPPER_POSITION_NORMALIZE_FN
from piper_constants import PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN
from utils import sample_cube_pose, sample_redbox_pose, sample_greenbox_pose, sample_bluebox_pose

import IPython
e = IPython.embed

REDBOX_POSE = [None] # to be changed from outside
BLUEBOX_POSE = [None]
GREENBOX_POSE = [None]
MANYCUBES_POSES = [None]

def make_sim_env(task_name, camera_names=None):
    """
    Environment for simulated robot bi-manual manipulation, with joint position control
    Action space:      [left_arm_qpos (6),             # absolute joint position
                        left_gripper_positions (1),    # normalized gripper position (0: close, 1: open)
                        right_arm_qpos (6),            # absolute joint position
                        right_gripper_positions (1),]  # normalized gripper position (0: close, 1: open)

    Observation space: {"qpos": Concat[ left_arm_qpos (6),         # absolute joint position
                                        left_gripper_position (1),  # normalized gripper position (0: close, 1: open)
                                        right_arm_qpos (6),         # absolute joint position
                                        right_gripper_qpos (1)]     # normalized gripper position (0: close, 1: open)
                        "qvel": Concat[ left_arm_qvel (6),         # absolute joint velocity (rad)
                                        left_gripper_velocity (1),  # normalized gripper velocity (pos: opening, neg: closing)
                                        right_arm_qvel (6),         # absolute joint velocity (rad)
                                        right_gripper_qvel (1)]     # normalized gripper velocity (pos: opening, neg: closing)
                        "images": {"main": (480x640x3)}        # h, w, c, dtype='uint8'
    """
    if 'sim_transfer_cube' in task_name:
        xml_path = os.path.join(XML_DIR, f'bimanual_piper_transfer_cube.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = TransferCubeTask(random=False, camera_names=camera_names)
        env = control.Environment(physics, task, time_limit=20, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    elif 'sim_insertion' in task_name:
        xml_path = os.path.join(XML_DIR, f'bimanual_piper_insertion.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = InsertionTask(random=False, camera_names=camera_names)
        env = control.Environment(physics, task, time_limit=20, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    elif 'sim_independent' in task_name:
        is_phase2 = 'phase2' in task_name
        xml_path = os.path.join(XML_DIR, f'bimanual_piper_many_cubes.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = ManyCubesTask(random=False, init_phase=2 if is_phase2 else 1, camera_names=camera_names)
        env = control.Environment(physics, task, time_limit=20, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    elif 'sim_coop' in task_name:
        is_phase2 = 'phase2' in task_name
        xml_path = os.path.join(XML_DIR, f'bimanual_piper_many_cubes.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = ManyCubesTask(random=False, init_phase=2 if is_phase2 else 1, camera_names=camera_names)
        env = control.Environment(physics, task, time_limit=20, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    elif 'sim_variable_coop' in task_name:
        xml_path = os.path.join(XML_DIR, f'bimanual_piper_variable_coop.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = ManyCubesTask(random=False, camera_names=camera_names)
        env = control.Environment(physics, task, time_limit=20, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    elif 'sim_many_cubes' in task_name:
        xml_path = os.path.join(XML_DIR, f'bimanual_piper_many_cubes.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = ManyCubesTask(random=False, camera_names=camera_names)
        env = control.Environment(physics, task, time_limit=20, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    else:
        raise NotImplementedError
    return env

class BimanualPiperTask(base.Task):
    def __init__(self, random=None, camera_names=None):
        super().__init__(random=random)
        self.belt_speed = 0  # デフォルトはベルトを停止
        self.camera_names = camera_names if camera_names is not None else ['top', 'angle', 'vis', 'l_wrist', 'r_wrist']

    def before_step(self, action, physics):
        # action: [left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)] = 14要素
        left_arm_action = action[0:6]
        normalized_left_gripper_action = action[6]
        right_arm_action = action[7:13]
        normalized_right_gripper_action = action[13]

        left_gripper_action = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(normalized_left_gripper_action)
        right_gripper_action = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(normalized_right_gripper_action)

        full_left_gripper_action = [left_gripper_action, -left_gripper_action]
        full_right_gripper_action = [right_gripper_action, -right_gripper_action]

        # env_action: [belt, left_arm(6), left_gripper(2), right_arm(6), right_gripper(2)] = 17要素
        env_action = np.concatenate([[self.belt_speed], left_arm_action, full_left_gripper_action, right_arm_action, full_right_gripper_action])

        super().before_step(env_action, physics)
        return
    

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        super().initialize_episode(physics)

    @staticmethod
    def get_qpos(physics):
        qpos_raw = physics.data.qpos.copy()
        # qpos 構造: [left_arm(6), left_gripper(2), right_arm(6), right_gripper(2), belt(1), boxes...]
        
        left_arm_qpos = qpos_raw[0:6]       # qpos[0:6]
        right_arm_qpos = qpos_raw[8:14]     # qpos[8:14]
        
        # gripper: qpos[6] を使う（対称制御なので片方だけ）
        left_gripper_qpos = [PUPPET_GRIPPER_POSITION_NORMALIZE_FN(qpos_raw[6])]
        right_gripper_qpos = [PUPPET_GRIPPER_POSITION_NORMALIZE_FN(qpos_raw[14])]
        
        # 出力順序: [left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)] = 14要素
        return np.concatenate([left_arm_qpos, left_gripper_qpos, right_arm_qpos, right_gripper_qpos])

    @staticmethod
    def get_qvel(physics):
        qvel_raw = physics.data.qvel.copy()
        # qvel も同じ構造
        
        left_arm_qvel = qvel_raw[0:6]
        right_arm_qvel = qvel_raw[8:14]
        
        left_gripper_qvel = [PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN(qvel_raw[6])]
        right_gripper_qvel = [PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN(qvel_raw[14])]
        
        return np.concatenate([left_arm_qvel, left_gripper_qvel, right_arm_qvel, right_gripper_qvel])

    @staticmethod
    def get_env_state(physics):
        raise NotImplementedError

    def get_observation(self, physics):
        obs = collections.OrderedDict()
        obs['qpos'] = self.get_qpos(physics)
        obs['qvel'] = self.get_qvel(physics)
        obs['env_state'] = self.get_env_state(physics)
        obs['images'] = dict()
        for cam_name in self.camera_names:
            if cam_name == 'top':
                obs['images']['top'] = physics.render(height=480, width=640, camera_id='top')
            elif cam_name == 'angle':
                obs['images']['angle'] = physics.render(height=480, width=640, camera_id='angle')
            elif cam_name == 'vis':
                obs['images']['vis'] = physics.render(height=480, width=640, camera_id='front_close')
            elif cam_name == 'l_wrist':
                obs['images']['l_wrist'] = physics.render(height=480, width=640, camera_id='l_wrist')
            elif cam_name == 'r_wrist':
                obs['images']['r_wrist'] = physics.render(height=480, width=640, camera_id='r_wrist')

        return obs

    def get_reward(self, physics):
        # return whether left gripper is holding the box
        raise NotImplementedError

class InsertionTask(BimanualPiperTask):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 4

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        # TODO Notice: this function does not randomize the env configuration. Instead, set BOX_POSE from outside
        # reset qpos, control and box position
        with physics.reset_context():
            physics.named.data.qpos[:16] = START_ARM_POSE
            np.copyto(physics.data.ctrl[:16], START_ARM_POSE)
            assert BOX_POSE[0] is not None
            physics.named.data.qpos[-7*2:] = BOX_POSE[0] # two objects
            # print(f"{BOX_POSE=}")
        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()[16:]
        return env_state

    def get_reward(self, physics):
        # return whether peg touches the pin
        all_contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, 'geom')
            name_geom_2 = physics.model.id2name(id_geom_2, 'geom')
            contact_pair = (name_geom_1, name_geom_2)
            all_contact_pairs.append(contact_pair)

        touch_right_gripper = ("red_peg", "right_gripper_1") in all_contact_pairs
        touch_left_gripper = ("socket-1", "left_gripper_1") in all_contact_pairs or \
                             ("socket-2", "left_gripper_1") in all_contact_pairs or \
                             ("socket-3", "left_gripper_1") in all_contact_pairs or \
                             ("socket-4", "left_gripper_1") in all_contact_pairs

        peg_touch_table = ("red_peg", "table") in all_contact_pairs
        socket_touch_table = ("socket-1", "table") in all_contact_pairs or \
                             ("socket-2", "table") in all_contact_pairs or \
                             ("socket-3", "table") in all_contact_pairs or \
                             ("socket-4", "table") in all_contact_pairs
        peg_touch_socket = ("red_peg", "socket-1") in all_contact_pairs or \
                           ("red_peg", "socket-2") in all_contact_pairs or \
                           ("red_peg", "socket-3") in all_contact_pairs or \
                           ("red_peg", "socket-4") in all_contact_pairs
        pin_touched = ("red_peg", "pin") in all_contact_pairs

        reward = 0
        if touch_left_gripper and touch_right_gripper: # touch both
            reward = 1
        if touch_left_gripper and touch_right_gripper and (not peg_touch_table) and (not socket_touch_table): # grasp both
            reward = 2
        if peg_touch_socket and (not peg_touch_table) and (not socket_touch_table): # peg and socket touching
            reward = 3
        if pin_touched: # successful insertion
            reward = 4
        return reward

class TransferCubeTask(BimanualPiperTask):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 4

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        # TODO Notice: this function does not randomize the env configuration. Instead, set BOX_POSE from outside
        # reset qpos, control and box position
        with physics.reset_context():
            physics.named.data.qpos[:16] = START_ARM_POSE
            
            # ctrl: belt がないので直接コピー
            np.copyto(physics.data.ctrl[:16], START_ARM_POSE)
            
            assert REDBOX_POSE[0] is not None
            physics.named.data.qpos[-7:] = REDBOX_POSE[0]

            for i in range(physics.model.nu):
                actuator_name = physics.model.actuator(i).name
                control_value = physics.data.ctrl[i]
                print(f"ctrl[{i}] -> Actuator '{actuator_name}': {control_value:.4f}")
                
        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()[16:]
        return env_state

    def get_reward(self, physics):
        # return whether left gripper is holding the box
        all_contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, 'geom')
            name_geom_2 = physics.model.id2name(id_geom_2, 'geom')
            contact_pair = (name_geom_1, name_geom_2)
            all_contact_pairs.append(contact_pair)

        touch_left_gripper = ("l_gripper_finger","red_box") in all_contact_pairs
        touch_right_gripper = ("r_gripper_finger","red_box") in all_contact_pairs
        touch_table = ("red_box", "table") in all_contact_pairs

        reward = 0
        if touch_right_gripper:
            reward = 1
        if touch_right_gripper and not touch_table: # lifted
            reward = 2
        if touch_left_gripper: # attempted transfer
            reward = 3
        if touch_left_gripper and not touch_table: # successful transfer
            reward = 4
        return reward
    
class MovingCubeTask(BimanualPiperTask):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 3
        self.belt_speed = BELT_MOVE_SPEED  # このタスクではベルトを動かす
        self.move_duration = 6.8 # seconds


    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        # TODO Notice: this function does not randomize the env configuration. Instead, set BOX_POSE from outside
        # reset qpos, control and box position
        with physics.reset_context():      
            physics.named.data.qpos[0:16] = START_ARM_POSE
            
            # ctrl への設定（belt を先頭に追加）
            # ctrl 構造: [belt(1), left_arm(6), left_gripper(2), right_arm(6), right_gripper(2)]
            ctrl_with_belt = np.concatenate([[self.belt_speed], START_ARM_POSE])
            np.copyto(physics.data.ctrl, ctrl_with_belt)
            
            assert REDBOX_POSE[0] is not None
            physics.named.data.qpos[-21:-14] = REDBOX_POSE[0]
            physics.named.data.qpos[-14:-7] = GREENBOX_POSE[0]
            physics.named.data.qpos[-7:] = BLUEBOX_POSE[0]

            for i in range(physics.model.nu):
                actuator_name = physics.model.actuator(i).name
                control_value = physics.data.ctrl[i]
                print(f"ctrl[{i}] -> Actuator '{actuator_name}': {control_value:.4f}")
                
        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()[17:17+21]
        return env_state

    def get_reward(self, physics):
        # return whether left gripper is holding the box
        all_contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, 'geom')
            name_geom_2 = physics.model.id2name(id_geom_2, 'geom')
            contact_pair = (name_geom_1, name_geom_2)
            all_contact_pairs.append(contact_pair)

        touch_right_gripper = ("r_gripper_finger","blue_box") in all_contact_pairs
        touch_left_gripper = ("l_gripper_finger","green_box") in all_contact_pairs
        green_touch_goal = ("goal_plate", "green_box") in all_contact_pairs
        blue_touch_goal = ("goal_plate", "blue_box") in all_contact_pairs
        touch_table = ("cushion1", "red_box") in all_contact_pairs or ("cushion1", "green_box") in all_contact_pairs or ("cushion1", "blue_box") in all_contact_pairs

        reward = 0
        if touch_right_gripper or touch_left_gripper:
            reward = 1
        if green_touch_goal or blue_touch_goal:
            reward = 2
        if green_touch_goal and blue_touch_goal: 
            reward = 3
        if touch_table:
            reward = 0

        return reward
    
class CoopTask(BimanualPiperTask):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 3
        self.belt_speed = BELT_MOVE_SPEED  # このタスクではベルトを動かす
        self.move_duration = 6.8 # seconds


    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        # TODO Notice: this function does not randomize the env configuration. Instead, set BOX_POSE from outside
        # reset qpos, control and box position
        with physics.reset_context():      
            physics.named.data.qpos[0:16] = START_ARM_POSE
            
            # ctrl への設定（belt を先頭に追加）
            # ctrl 構造: [belt(1), left_arm(6), left_gripper(2), right_arm(6), right_gripper(2)]
            ctrl_with_belt = np.concatenate([[self.belt_speed], START_ARM_POSE])
            np.copyto(physics.data.ctrl, ctrl_with_belt)
            
            assert REDBOX_POSE[0] is not None
            physics.named.data.qpos[-21:-14] = REDBOX_POSE[0]
            physics.named.data.qpos[-14:-7] = GREENBOX_POSE[0]
            physics.named.data.qpos[-7:] = BLUEBOX_POSE[0]

            for i in range(physics.model.nu):
                actuator_name = physics.model.actuator(i).name
                control_value = physics.data.ctrl[i]
                print(f"ctrl[{i}] -> Actuator '{actuator_name}': {control_value:.4f}")
                
        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()[17:17+21]
        return env_state


    def get_reward(self, physics):
        # return whether left gripper is holding the box
        all_contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, 'geom')
            name_geom_2 = physics.model.id2name(id_geom_2, 'geom')
            contact_pair = (name_geom_1, name_geom_2)
            all_contact_pairs.append(contact_pair)

        touch_right_gripper = ("r_gripper_finger","cube_9") in all_contact_pairs
        touch_left_gripper = ("l_gripper_finger","cube_8") in all_contact_pairs
        assembled = ("cube_8", "cube_9") in all_contact_pairs
        touch_goal_area = ("goal_plate", "cube_9") in all_contact_pairs
        touch_table = ("cushion1", "cube_8") in all_contact_pairs or ("cushion1", "cube_9") in all_contact_pairs or ("cushion1", "cube_7") in all_contact_pairs
        touch_red_box = ("l_gripper_finger", "cube_7") in all_contact_pairs
        placed_red_box = ("goal_plate", "cube_7") in all_contact_pairs
        reward = 0
        if touch_right_gripper or touch_left_gripper:
            reward = 1
        if assembled :
            reward = 2
        if touch_goal_area and assembled: 
            reward = 3
        if touch_red_box and placed_red_box:
            reward = 4
        if touch_table:
            reward = 0

        return reward


class ManyCubesTask(BimanualPiperTask):
    def __init__(self, random=None, randomize_cube_colors=False, init_phase=1, camera_names=None):
        super().__init__(random=random, camera_names=camera_names)
        self.max_reward = 4
        self.belt_speed = BELT_MOVE_SPEED
        self.randomize_cube_colors = randomize_cube_colors
        self.init_phase = init_phase

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        with physics.reset_context():
            physics.named.data.qpos[0:16] = START_ARM_POSE
            
            ctrl_with_belt = np.concatenate([[self.belt_speed], START_ARM_POSE])
            np.copyto(physics.data.ctrl, ctrl_with_belt)
            
            # Start position and spacing for queue
            start_x = 0.2
            spacing = -0.15 
            
            # Colors: R, G, B
            colors = [
                np.array([1, 0, 0, 1]), # R
                np.array([0, 1, 0, 1]), # G
                np.array([0, 0, 1, 1])  # B
            ]
            
            # Request: B, G, R from right. 
            # Rightmost is max index (i=9) in our loop.
            # i=9 -> Blue (Coop Range)
            # i=8 -> Green (Coop Range)
            # i=7 -> Red (Coop Range)
            # i=6..0 -> Queue behind Red
            
            poses = {}
            
            if self.init_phase == 2:
                # Phase 2: G/B at goal, R and Queue at ORIGINAL positions
                
                if MANYCUBES_POSES[0] is not None:
                    # Injected poses
                    poses = MANYCUBES_POSES[0]
                else:
                    # Sample new poses
                    # Green (8) at goal + noise
                    # noise_range: 2cm
                    noise_g = np.random.uniform(-0.02, 0.02, size=2)
                    poses[8] = np.array([0 + noise_g[0], 0.1 + noise_g[1], 0.025, 1, 0, 0, 0])
                    # Blue (9) at goal + noise
                    noise_b = np.random.uniform(-0.02, 0.02, size=2)
                    poses[9] = np.array([0.08 + noise_b[0], 0.1 + noise_b[1], 0.025, 1, 0, 0, 0])
                    # Red (7) at original
                    poses[7] = sample_redbox_pose()+np.array([0.2, 0, 0,0,0,0,0])
                ref_x = poses[7][0]
                
            else:
                # Phase 1: Normal
                if BLUEBOX_POSE[0] is not None:
                    poses[9] = BLUEBOX_POSE[0]
                else:
                    poses[9] = sample_bluebox_pose()

                if GREENBOX_POSE[0] is not None:
                    poses[8] = GREENBOX_POSE[0]
                else:
                    poses[8] = sample_greenbox_pose()

                if REDBOX_POSE[0] is not None:
                    poses[7] = REDBOX_POSE[0]
                else:
                    poses[7] = sample_redbox_pose()
                
                # Fix: Anchor queue to the LEFT-MOST active cube (min X)
                # to prevent queue overlapping with other cubes if Red is far right.
                xs = [poses[7][0], poses[8][0], poses[9][0]]
                ref_x = min(xs)
            
            queue_spacing = 0.22
            
            for i in range(10):
                if i in poses:
                    cube_pose = poses[i]
                else:
                    # i=6 -> 1 step behind 7 (Original)
                    step = 7 - i
                    # Original pos would be: ref_x - step * spacing
                    orig_x = ref_x - step * queue_spacing 
                    
                    cube_x = orig_x + np.random.uniform(-0.01, 0.01)
                        
                    cube_y = np.random.uniform(0.30, 0.45) 
                    
                    cube_quat = np.array([1, 0, 0, 0])
                    cube_pose = np.concatenate([[cube_x, cube_y, 0.02], cube_quat])

                start_idx = physics.model.name2id(f'cube_{i}_joint', 'joint')
                qpos_adr = physics.model.jnt_qposadr[start_idx]
                np.copyto(physics.data.qpos[qpos_adr : qpos_adr + 7], cube_pose)
                
                # Color logic
                if self.randomize_cube_colors:
                    color = colors[np.random.randint(0, 3)]
                else:
                    color_idx = (i + 2) % 3
                    color = colors[color_idx]

                geom_id = physics.model.name2id(f'cube_{i}', 'geom')
                physics.model.geom_rgba[geom_id] = color

            for i in range(physics.model.nu):
                actuator_name = physics.model.actuator(i).name
                control_value = physics.data.ctrl[i]
                # print(f"ctrl[{i}] -> Actuator '{actuator_name}': {control_value:.4f}")
                
        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        # return state of 10 cubes (each 7 dims) -> 70 dims
        # qpos structure: robot (16) + belt (1) + belt_extension (1) + 10 cubes (7*10)
        # Total non-cube joints = 18.
        env_state = physics.data.qpos.copy()[18:18+70]
        return env_state

    def get_reward(self, physics):
        # return whether left gripper is holding the box
        all_contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, 'geom')
            name_geom_2 = physics.model.id2name(id_geom_2, 'geom')
            contact_pair = (name_geom_1, name_geom_2)
            all_contact_pairs.append(contact_pair)

        touch_right_gripper = ("r_gripper_finger","cube_9") in all_contact_pairs
        touch_left_gripper = ("l_gripper_finger","cube_8") in all_contact_pairs
        assembled = ("cube_8", "cube_9") in all_contact_pairs
        touch_goal_area = ("goal_plate", "cube_9") in all_contact_pairs
        touch_table = ("cushion1", "cube_8") in all_contact_pairs or ("cushion1", "cube_9") in all_contact_pairs or ("cushion1", "cube_7") in all_contact_pairs
        touch_red_box = ("l_gripper_finger", "cube_7") in all_contact_pairs
        placed_red_box = ("goal_plate", "cube_7") in all_contact_pairs
        reward = 0
        if touch_right_gripper or touch_left_gripper:
            reward = 1
        if assembled :
            reward = 2
        if touch_goal_area and assembled: 
            reward = 3
        if placed_red_box:
            reward = 4
        if touch_table:
            reward = 0

        return reward


def get_action(master_bot_left, master_bot_right):
    action = np.zeros(14)
    # arm action
    action[:6] = master_bot_left.dxl.joint_states.position[:6]
    action[7:7+6] = master_bot_right.dxl.joint_states.position[:6]
    # gripper action
    left_gripper_pos = master_bot_left.dxl.joint_states.position[7]
    right_gripper_pos = master_bot_right.dxl.joint_states.position[7]
    normalized_left_pos = MASTER_GRIPPER_POSITION_NORMALIZE_FN(left_gripper_pos)
    normalized_right_pos = MASTER_GRIPPER_POSITION_NORMALIZE_FN(right_gripper_pos)
    action[6] = normalized_left_pos
    action[7+6] = normalized_right_pos
    return action

def test_sim_teleop():
    """ Testing teleoperation in sim with ALOHA. Requires hardware and ALOHA repo to work. """
    from interbotix_xs_modules.arm import InterbotixManipulatorXS

    BOX_POSE[0] = [0.2, 0.5, 0.05, 1, 0, 0, 0]

    # source of data
    master_bot_left = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                              robot_name=f'master_left', init_node=True)
    master_bot_right = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                              robot_name=f'master_right', init_node=False)

    # setup the environment
    env = make_sim_env('sim_transfer_cube')
    ts = env.reset()
    episode = [ts]
    # setup plotting
    ax = plt.subplot()
    plt_img = ax.imshow(ts.observation['images']['angle'])
    plt.ion()

    for t in range(1000):
        action = get_action(master_bot_left, master_bot_right)
        ts = env.step(action)
        episode.append(ts)

        plt_img.set_data(ts.observation['images']['angle'])
        plt.pause(0.02)

if __name__ == '__main__':
    from utils import sample_redbox_pose, sample_greenbox_pose, sample_bluebox_pose, sample_cube_pose
    
    REDBOX_POSE[0] = sample_redbox_pose()
    GREENBOX_POSE[0] = sample_greenbox_pose()
    BLUEBOX_POSE[0] = sample_bluebox_pose()
    
    env = make_sim_env('sim_coop')
    # テスト用にベルトを停止
    env._task.belt_speed = 0
    
    ts = env.reset()
    physics = env.physics

    # アクチュエータのゲイン確認
    print("\n=== Actuator gains ===")
    for i in range(physics.model.nu):
        actuator_name = physics.model.actuator(i).name
        kp = physics.model.actuator_gainprm[i, 0]
        print(f"Actuator[{i}] '{actuator_name}': kp={kp}")

    print("\n=== qpos 構造確認 ===")
    print(f"qpos[0:6] (left_arm): {physics.data.qpos[0:6]}")
    print(f"qpos[6] (left_gripper): {physics.data.qpos[6]}")
    print(f"qpos[8:14] (right_arm): {physics.data.qpos[8:14]}")
    print(f"qpos[14] (right_gripper): {physics.data.qpos[14]}")
    
    print("\n=== get_qpos 出力 ===")
    qpos_out = BimanualPiperTask.get_qpos(physics)
    print(f"Shape: {qpos_out.shape}")
    print(f"Values: {qpos_out}")
    
    print("\n=== action と qpos の対応確認 ===")
    # get_qpos で取得した値をそのまま action として使う
    action = qpos_out.copy()
    print(f"action[0:6] (left_arm): {action[0:6]}")
    print(f"action[6] (left_gripper): {action[6]}")
    print(f"action[7:13] (right_arm): {action[7:13]}")
    print(f"action[13] (right_gripper): {action[13]}")
    
    # 複数ステップ実行して安定性を確認
    print("\n=== 複数ステップでの qpos 追跡 ===")
    for step in range(10):
        ts = env.step(action)
        qpos_after = BimanualPiperTask.get_qpos(physics)
        diff = np.abs(qpos_out - qpos_after)
        max_diff = diff.max()
        max_idx = diff.argmax()
        print(f"Step {step+1}: 最大差分 = {max_diff:.6f} (index={max_idx})")
        if max_diff > 0.01:
            print(f"  差分詳細: {diff}")
    
    qpos_after = BimanualPiperTask.get_qpos(physics)
    print("\n=== 最終 step 後の qpos ===")
    print(f"差分: {np.abs(qpos_out - qpos_after).max():.6f}")
    if np.abs(qpos_out - qpos_after).max() < 0.01:
        print("✓ OK: qpos が正しく維持されています")
    else:
        print("✗ NG: qpos が正しく維持されていません")
        print("\n左アーム差分:")
        left_diff = np.abs(qpos_out[0:6] - qpos_after[0:6])
        for i, d in enumerate(left_diff):
            print(f"  joint{i+1}: {d:.6f}")
        print("\n右アーム差分:")
        right_diff = np.abs(qpos_out[7:13] - qpos_after[7:13])
        for i, d in enumerate(right_diff):
            print(f"  joint{i+1}: {d:.6f}")