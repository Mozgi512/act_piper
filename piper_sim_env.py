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
from piper_constants import MANYCUBES_COLORS, MANYCUBES_CONFIG
from utils import sample_cube_pose, sample_redbox_pose, sample_greenbox_pose, sample_bluebox_pose

import IPython
e = IPython.embed

REDBOX_POSE = [None] # to be changed from outside
BLUEBOX_POSE = [None]
GREENBOX_POSE = [None]
MANYCUBES_POSES = [None]
# MANYCUBES_COLORS imported directly now
MANYCUBES_TASK_COUNT = [None]  # Expected number of tasks to complete

def make_sim_env(task_name, camera_names=None, time_limit=20, interleave_last_four=False):
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
        env = control.Environment(physics, task, time_limit=time_limit, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    elif 'sim_coop' in task_name:
        is_phase2 = 'phase2' in task_name
        xml_path = os.path.join(XML_DIR, f'bimanual_piper_many_cubes.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = ManyCubesTask(random=False, init_phase=2 if is_phase2 else 1, camera_names=camera_names)
        env = control.Environment(physics, task, time_limit=time_limit, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    elif 'sim_variable_coop' in task_name:
        xml_path = os.path.join(XML_DIR, f'bimanual_piper_variable_coop.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = ManyCubesTask(random=False, camera_names=camera_names)
        env = control.Environment(physics, task, time_limit=time_limit, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    elif 'sim_four_objects' in task_name:
        xml_path = os.path.join(XML_DIR, f'bimanual_piper_many_cubes.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = ManyCubesTask(random=False, camera_names=camera_names)
        env = control.Environment(physics, task, time_limit=time_limit, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    elif 'sim_many_cubes' in task_name:
        xml_path = os.path.join(XML_DIR, f'bimanual_piper_many_cubes.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = ManyCubesTask(random=False, camera_names=camera_names, interleave_last_four=interleave_last_four)
        env = control.Environment(physics, task, time_limit=time_limit, control_timestep=DT,
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
                obs['images']['top'] = physics.render(height=240, width=320, camera_id='top')
            elif cam_name == 'angle':
                obs['images']['angle'] = physics.render(height=240, width=320, camera_id='angle')
            elif cam_name == 'vis':
                obs['images']['vis'] = physics.render(height=240, width=320, camera_id='front_close')
            elif cam_name == 'l_wrist':
                obs['images']['l_wrist'] = physics.render(height=240, width=320, camera_id='l_wrist')
            elif cam_name == 'r_wrist':
                obs['images']['r_wrist'] = physics.render(height=240, width=320, camera_id='r_wrist')

        return obs

    def get_reward(self, physics):
        # return whether left gripper is holding the box
        raise NotImplementedError

class InsertionTask(BimanualPiperTask):
    def __init__(self, random=None, camera_names=None):
        super().__init__(random=random, camera_names=camera_names)
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
    def __init__(self, random=None, camera_names=None):
        super().__init__(random=random, camera_names=camera_names)
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
    def __init__(self, random=None, randomize_cube_colors=False, init_phase=1, camera_names=None, interleave_last_four=False):
        super().__init__(random=random, camera_names=camera_names)
        self.interleave_last_four = interleave_last_four
        self.max_reward = 4  # Will be updated dynamically based on color sequence
        self.belt_speed = BELT_MOVE_SPEED
        self.randomize_cube_colors = randomize_cube_colors
        self.init_phase = init_phase
        self.color_sequence = None  # Will be set from MANYCUBES_COLORS during initialize_episode
        # Track completed tasks cumulatively (survives object removal)
        self.completed_independent_cubes = set()  # Set of cube_i indices
        self.completed_cooperative_pairs = set()  # Set of (green_idx, blue_idx) tuples

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        with physics.reset_context():
            physics.named.data.qpos[0:16] = START_ARM_POSE
            
            ctrl_with_belt = np.concatenate([[self.belt_speed], START_ARM_POSE])
            np.copyto(physics.data.ctrl, ctrl_with_belt)
            
            # Start position and spacing for queue
            start_x = 0.0
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
            
            if MANYCUBES_POSES[0] is not None:
                poses = MANYCUBES_POSES[0]
                # Dummy ref_x, not used if all poses provided
                ref_x = 0
            elif self.init_phase == 2:
                # Phase 2: G/B at goal, R and Queue at ORIGINAL positions
                
                if MANYCUBES_POSES[0] is not None:
                    # Injected poses
                    poses = MANYCUBES_POSES[0]
                else:
                    # Sample new poses
                    # Green (8) at goal + noise
                    # noise_range: 2cm
                    noise_g = np.random.uniform(-0.05, 0.05, size=2)
                    poses[8] = np.array([0 + noise_g[0], 0.1 + noise_g[1], 0.025, 1, 0, 0, 0])
                    # Blue (9) at goal + noise
                    noise_b = np.random.uniform(-0.05, 0.05, size=2)
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

                # Interactive Mode Override: Linear Queue 0..9
                # 0 is First (Rightmost on belt start), 9 is Last (Back of queue)
                # But physically belt moves +X to -X.
                # If we want 0 to arrive FIRST, 0 should be Left-most (closest to center) and 9 Right-most.
                # Current logic: Belt moves +X -> -X??
                # Wait, BELT_MOVE_SPEED is usually positive?
                # If positive, does it move +X or -X?
                # piper_constants.py: BELT_MOVE_SPEED = 0.005?
                # In simulation xml, belt direction?
                # If items spawn at 0.4 and drift to -0.5, belt moves -X.
                # So Upstream is +X. Downstream is -X.
                # Order 0, 1, 2...
                # 0 should be Downstream (First to pick).
                # 1 should be Upstream of 0.
                if MANYCUBES_COLORS[0] is not None:
                     poses = {} # Clear specials
                     start_x = 0.0 # Start of working area
                     spacing = 0.15 # Spacing towards Upstream (+X)
                     for i in range(10):
                         # x = start + i*spacing
                         # 0: 0.35
                         # 1: 0.53
                         # ...
                         px = start_x - i * spacing
                         # Apply global shift
                         shift_val = MANYCUBES_CONFIG.get('x_shift', 0.0)
                         px += shift_val
                         
                         #if i == 3:
                             #px -= 0.02
                         #if i == 1:
                             #px -= 0.02
                         #if i == 5:
                             #px += 0.02
                         #if i == 8:
                             #px -= 0.05  
                         #if i == 9:
                             #px -= 0.05
                         # Apply X Randomization
                         px += np.random.uniform(-0.04, 0.04)
                         py = np.random.uniform(0.32, 0.45)

                         # Check for target_indices filter
                         target_indices = MANYCUBES_CONFIG.get('target_indices')
                         if target_indices is not None and i not in target_indices:
                             # Hide non-target object
                             poses[i] = np.array([10.0 + i, -10.0, -1.0, 1, 0, 0, 0])
                         else:
                             poses[i] = np.array([px, py, 0.025, 1, 0, 0, 0])
                             print(f"Debug: Cube {i} initialized at X={px:.3f}, Y={py:.3f}")
            
                     ref_x = 0 # Ignored
            
            queue_spacing = 0.22
            
            # Scramble indices if requested
            # Normal: 0,1,2,3,4,5,6,7,8,9 (0 is downstream/closest)
            # Interleave: 0, 6, 1, 7, 2, 8, 3, 9, 4, 5
            slot_to_cube_map = list(range(10))
            if self.interleave_last_four:
                print("DEBUG: Interleaving last 4 objects among first 5")
                slot_to_cube_map = [0, 6, 1, 7, 2, 8, 3, 9, 4, 5]

            for slot_i in range(10):
                # The cube index we are placing at this spatial slot
                i = slot_to_cube_map[slot_i]

                if i in poses:
                    cube_pose = poses[i]
                else:
                    # slot_i determines position relative to start
                    # i=6 -> 1 step behind 7 (Original logic was weird because it used 'i' for position)
                    # New logic: Use slot_i for spacing
                    
                    # Original logic used fixed ref_x and 'step = 7 - i'. 
                    # If we use slot_i, we just place them sequentially.
                    # Start X = ref_x
                    # X = ref_x - slot_i * queue_spacing (if moving +X to -X???)
                    # Wait, original logic:
                    # i=7 (Red) is ref. i=6 is 1 step behind.
                    # so i=6 X > i=7 X ???
                    # If belt moves -X (items travel Left), then Upstream is Right.
                    # If 6 is "Behind" 7, it means 6 arrives LATER. So 6 is Right of 7.
                    # 7 is Downstream. 6 is Upstream.
                    # step = 7 - 6 = 1.
                    # orig_x = ref_x - step * spacing.
                    # if spacing is negative? No.
                    # Let's check: spacing = -0.15 (line 455). queue_spacing = 0.22.
                    # ref_x was min(xs) (leftmost).
                    # This logic is extremely messy in original code.
                    
                    # SIMPLIFIED LOGIC for INTERACTIVE/EVAL MODE (where MANYCUBES_COLORS is set)
                        # We already set poses in line 536 loop!
                        # BUT wait, the loop 536 sets poses for 'i' in range(10).
                        # If we interleave, we want CUBE 6 to be at SLOT 1 position.
                        # So we should re-assign based on slot map.
                        
                    pass
                    
                    # If poses[i] is NOT set (e.g. queue logic without colors?)
                    # Fallback to slot-based placement
                    # We assume slot 0 is closest/first.
                    # For safe placement, let's just use the override loop above if colors set.
                    
                    # Re-verify the loop 536-545:
                    # So if poses[i] implies "Position for Cube i", then Cube 6 is at Pos 6.
                    # We want Cube 6 at Pos 1.
                    
                    # So we must modify the PREVIOUS loop (line 536) or this one.
                    # Easier to modify the PREVIOUS loop (generation).
                    pass

            # REDO GENERATION LOGIC for Interleave (Override previous loop 536)
            if MANYCUBES_COLORS[0] is not None and self.interleave_last_four:
                 start_x = 0.00
                 # DENSE PLACEMENT: Reduce spacing to 0.08 (approx half of 0.15)
                 # effectively filling the gaps between typical 0.15 spacing.
                 # Cube size is ~0.05, so 0.08 leaves 0.03 gap.
                 spacing = 0.08 
                 
                 # slot_to_cube_map defined above
                 for slot_i in range(10):
                     cube_idx = slot_to_cube_map[slot_i]
                     px = start_x - slot_i * spacing
                     # Apply global shift
                     px += MANYCUBES_CONFIG.get('x_shift', 0.0)

                     # Apply X Randomization
                     px += np.random.uniform(-0.05, 0.05)
                     py = np.random.uniform(0.30, 0.50)
                     
                     # Check for target_indices filter
                     target_indices = MANYCUBES_CONFIG.get('target_indices')
                     if target_indices is not None and cube_idx not in target_indices:
                         # Hide non-target object
                         poses[cube_idx] = np.array([10.0 + cube_idx, -10.0, -1.0, 1, 0, 0, 0])
                     else:
                         poses[cube_idx] = np.array([px, py, 0.025, 1, 0, 0, 0])
                         print(f"DEBUG: Interleave Dense - Cube {cube_idx} at Slot {slot_i} (X={px:.3f})")

            for i in range(10):
                # Standard application loop
                if i in poses:
                    cube_pose = poses[i]
                else:
                    # ... legacy queue logic ...
                    # If this runs, it means i wasn't in poses.
                    # But for eval mode, all 10 are in poses.
                    # Safe to ignore legacy branch for now.
                    step = 7 - i
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
                    # Check for injected color sequence
                    if MANYCUBES_COLORS[0] is not None and i < len(MANYCUBES_COLORS[0]):
                        c_code = MANYCUBES_COLORS[0][i]
                        if c_code == 'r': color = colors[0]
                        elif c_code == 'g': color = colors[1]
                        elif c_code == 'b': color = colors[2]
                        else: color = colors[0] # Default Red
                    
                    # Explicit mapping for Four Object Task (Legacy/Fallback)
                    elif i == 6 or i == 7: # Red
                        color = colors[0]
                    elif i == 8: # Green
                        color = colors[1]
                    elif i == 9: # Blue
                        color = colors[2]
                    else:
                        # Fallback for others (queue)
                        color_idx = (i + 2) % 3
                        color = colors[color_idx]

                geom_id = physics.model.name2id(f'cube_{i}', 'geom')
                physics.model.geom_rgba[geom_id] = color

            for i in range(physics.model.nu):
                actuator_name = physics.model.actuator(i).name
                control_value = physics.data.ctrl[i]
                # print(f"ctrl[{i}] -> Actuator '{actuator_name}': {control_value:.4f}")
            
            # Store color sequence for reward calculation
            if MANYCUBES_COLORS[0] is not None:
                self.color_sequence = MANYCUBES_COLORS[0]
                # Set max_reward based on expected task count (if provided)
                # Set max_reward based on color sequence
                # Ignore MANYCUBES_TASK_COUNT for max_reward value since it doesn't account for weights
                red_count = self.color_sequence.count('r')
                green_count = self.color_sequence.count('g')
                blue_count = self.color_sequence.count('b')
                independent_pairs = red_count # Individual red tasks
                cooperative_pairs = min(green_count, blue_count)
                self.max_reward = 1 * independent_pairs + 2 * cooperative_pairs
            
            # Reset completed task tracking
            self.completed_independent_cubes = set()
            self.completed_cooperative_pairs = set()
                
        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        # return state of 10 cubes (each 7 dims) -> 70 dims
        # qpos structure: robot (16) + belt (1) + belt_extension (1) + 10 cubes (7*10)
        # Total non-cube joints = 18.
        env_state = physics.data.qpos.copy()[18:18+70]
        return env_state

    def get_reward(self, physics):
        """
        Flexible reward calculation with cumulative tracking:
        - Independent task (RR): 2 Red cubes in goal zone = +1 (tracked cumulatively)
        - Cooperative task (GB): 1 assembled Green+Blue in goal zone = +1 (tracked cumulatively)
        Returns total number of completed task pairs (cumulative).
        """
        # If no color sequence set, fall back to old logic
        if self.color_sequence is None:
            return self._get_reward_legacy(physics)
        
        # Get all contact pairs
        all_contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, 'geom')
            name_geom_2 = physics.model.id2name(id_geom_2, 'geom')
            all_contact_pairs.append((name_geom_1, name_geom_2))
        
        # Find cubes by color and check goal/assembly status
        red_in_goal = []
        green_cubes = []  # All green cubes (may or may not be in goal)
        blue_in_goal = []  # Blue cubes that ARE in goal
        
        for i in range(10):
            cube_name = f'cube_{i}'
            color = self.color_sequence[i]
            in_goal = (('goal_plate', cube_name) in all_contact_pairs or 
                      (cube_name, 'goal_plate') in all_contact_pairs)
            
            if color == 'r' and in_goal:
                red_in_goal.append(i)
            elif color == 'g':
                green_cubes.append(i)  # Track all greens, not just in goal
            elif color == 'b' and in_goal:
                blue_in_goal.append(i)  # Only blues in goal
        
        # Mark new Independent items as completed
        for i in red_in_goal:
            if i not in self.completed_independent_cubes:
                self.completed_independent_cubes.add(i)
                print(f"  [Reward] Completed Independent task: R{i}")

        # Mark new Cooperative pairs as completed
        # Success condition: G-B assembled AND B in goal (G doesn't need to be in goal)
        used_blues = set()
        for g_idx in green_cubes:  # Check ALL green cubes
            for b_idx in blue_in_goal:  # Only blues that are in goal
                if b_idx in used_blues:
                    continue
                # Check if green and blue are assembled (in contact OR close proximity)
                is_contact = ((f'cube_{g_idx}', f'cube_{b_idx}') in all_contact_pairs or
                             (f'cube_{b_idx}', f'cube_{g_idx}') in all_contact_pairs)
                
                is_close = False
                if not is_contact:
                    try:
                        g_body_id = physics.model.name2id(f'cube_{g_idx}', 'body')
                        b_body_id = physics.model.name2id(f'cube_{b_idx}', 'body')
                        g_pos = physics.data.xpos[g_body_id]
                        b_pos = physics.data.xpos[b_body_id]
                        dist = np.linalg.norm(g_pos - b_pos)
                        if dist < 0.10: # 10cm threshold to capture visual stacking
                            is_close = True
                    except: pass

                if is_contact or is_close:
                    pair = (g_idx, b_idx)
                    if pair not in self.completed_cooperative_pairs:
                        self.completed_cooperative_pairs.add(pair)
                        print(f"  [Reward] Completed Cooperative pair: G{g_idx} + B{b_idx} (assembled, B in goal)")
                    used_blues.add(b_idx)
                    break  # Each green can only pair once
        
        # Return total completed tasks
        total_reward = 1 * len(self.completed_independent_cubes) + 2 * len(self.completed_cooperative_pairs)
        return total_reward
    
    def _get_reward_legacy(self, physics):
        """Legacy reward calculation for backward compatibility"""
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