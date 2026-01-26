import numpy as np
import collections
import os

from piper_constants import DT, XML_DIR, START_ARM_POSE,BELT_MOVE_SPEED
from piper_constants import PUPPET_GRIPPER_POSITION_CLOSE
from piper_constants import PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN
from piper_constants import PUPPET_GRIPPER_POSITION_NORMALIZE_FN
from piper_constants import PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN

from utils import sample_redbox_pose, sample_insertion_pose,sample_bluebox_pose,sample_greenbox_pose,sample_cube_pose
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base

import IPython
e = IPython.embed


def make_ee_sim_env(task_name, camera_names=None):
    """
    Environment for simulated robot bi-manual manipulation, with end-effector control.
    Action space:      [left_arm_pose (7),             # position and quaternion for end effector
                        left_gripper_positions (1),    # normalized gripper position (0: close, 1: open)
                        right_arm_pose (7),            # position and quaternion for end effector
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
        xml_path = os.path.join(XML_DIR, f'bimanual_piper_ee_transfer_cube.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = TransferCubeEETask(random=False, camera_names=camera_names)
        env = control.Environment(physics, task, time_limit=20, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    elif 'sim_insertion' in task_name:
        xml_path = os.path.join(XML_DIR, f'bimanual_piper_ee_insertion.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = InsertionEETask(random=False, camera_names=camera_names)
        env = control.Environment(physics, task, time_limit=20, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    elif 'sim_independent' in task_name:
        is_phase2 = 'phase2' in task_name
        xml_path = os.path.join(XML_DIR, f'bimanual_piper_ee_many_cubes.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = ManyCubesEETask(random=False, init_phase=2 if is_phase2 else 1, camera_names=camera_names)
        env = control.Environment(physics, task, time_limit=20, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    elif 'sim_coop' in task_name:
        xml_path = os.path.join(XML_DIR, f'bimanual_piper_ee_many_cubes.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = ManyCubesEETask(random=False, camera_names=camera_names)
        env = control.Environment(physics, task, time_limit=20, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    elif 'sim_variable_coop' in task_name:
        xml_path = os.path.join(XML_DIR, f'bimanual_piper_ee_variable_coop.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = ManyCubesEETask(random=False, camera_names=camera_names)
        env = control.Environment(physics, task, time_limit=20, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    elif 'sim_many_cubes' in task_name:
        xml_path = os.path.join(XML_DIR, f'bimanual_piper_ee_many_cubes.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = ManyCubesEETask(random=False, camera_names=camera_names)
        env = control.Environment(physics, task, time_limit=20, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    else:
        raise NotImplementedError
    return env

class BimanualPiperEETask(base.Task):
    def __init__(self, random=None, camera_names=None):
        super().__init__(random=random)
        self.camera_names = camera_names if camera_names is not None else ['top', 'angle', 'vis', 'l_wrist', 'r_wrist']

    def before_step(self, action, physics):
        a_len = (len(action) -7)// 2
        action_left = action[:a_len]
        action_right = action[a_len:]
        # set mocap position and quat
        # left
        np.copyto(physics.data.mocap_pos[0], action_left[:3])
        np.copyto(physics.data.mocap_quat[0], action_left[3:7])
        # right
        np.copyto(physics.data.mocap_pos[1], action_right[:3])
        np.copyto(physics.data.mocap_quat[1], action_right[3:7])

        # set gripper
        g_left_ctrl = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(action_left[7])
        g_right_ctrl = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(action_right[7])
        np.copyto(physics.data.ctrl[1:3], np.array([g_left_ctrl,g_right_ctrl]))

    def initialize_robots(self, physics):
        # reset joint position
        physics.named.data.qpos[:16] = START_ARM_POSE

        # reset mocap to align with end effector
        # to obtain these numbers:
        # (1) make an ee_sim env and reset to the same start_pose
        # (2) get env._physics.named.data.xpos['vx300s_left/gripper_link']
        #     get env._physics.named.data.xquat['vx300s_left/gripper_link']
        #     repeat the same for right side
        np.copyto(physics.data.mocap_pos[0], [-0.1, 0.1, 0.2])
        np.copyto(physics.data.mocap_quat[0], [1, 0, 0, 0])
        # right
        np.copyto(physics.data.mocap_pos[1], np.array([0.1, 0.1, 0.2]))
        np.copyto(physics.data.mocap_quat[1],  [1, 0, 0, 0])

        # reset gripper control
        close_gripper_control = np.array([
            PUPPET_GRIPPER_POSITION_CLOSE,
            PUPPET_GRIPPER_POSITION_CLOSE
        ])
        np.copyto(physics.data.ctrl[1:3], close_gripper_control)

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        super().initialize_episode(physics)

    @staticmethod
    def get_qpos(physics):
        qpos_raw = physics.data.qpos.copy()
        left_qpos_raw = qpos_raw[:8]
        right_qpos_raw = qpos_raw[8:16]
        left_arm_qpos = left_qpos_raw[:6]
        right_arm_qpos = right_qpos_raw[:6]
        left_gripper_qpos = [PUPPET_GRIPPER_POSITION_NORMALIZE_FN(left_qpos_raw[6])]
        right_gripper_qpos = [PUPPET_GRIPPER_POSITION_NORMALIZE_FN(right_qpos_raw[6])]
        return np.concatenate([left_arm_qpos, left_gripper_qpos, right_arm_qpos, right_gripper_qpos])

    @staticmethod
    def get_qvel(physics):
        qvel_raw = physics.data.qvel.copy()
        left_qvel_raw = qvel_raw[:8]
        right_qvel_raw = qvel_raw[8:16]
        left_arm_qvel = left_qvel_raw[:6]
        right_arm_qvel = right_qvel_raw[:6]
        left_gripper_qvel = [PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN(left_qvel_raw[6])]
        right_gripper_qvel = [PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN(right_qvel_raw[6])]
        return np.concatenate([left_arm_qvel, left_gripper_qvel, right_arm_qvel, right_gripper_qvel])

    @staticmethod
    def get_env_state(physics):
        raise NotImplementedError

    def get_observation(self, physics):
        # note: it is important to do .copy()
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
        # used in scripted policy to obtain starting pose
        obs['mocap_pose_left'] = np.concatenate([physics.data.mocap_pos[0], physics.data.mocap_quat[0]]).copy()
        obs['mocap_pose_right'] = np.concatenate([physics.data.mocap_pos[1], physics.data.mocap_quat[1]]).copy()
        obs['belt_state'] = physics.data.qpos[16].copy()
        obs['gripper_ctrl'] = physics.data.ctrl[1:3].copy()
        return obs

    def get_reward(self, physics):
        raise NotImplementedError
    
class InsertionEETask(BimanualPiperEETask):
    def __init__(self, random=None, camera_names=None):
        super().__init__(random=random, camera_names=camera_names)
        self.max_reward = 4

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        self.initialize_robots(physics)
        # randomize peg and socket position
        peg_pose, socket_pose = sample_insertion_pose()
        id2index = lambda j_id: 16 + (j_id - 16) * 7 # first 16 is robot qpos, 7 is pose dim # hacky

        peg_start_id = physics.model.name2id('red_peg_joint', 'joint')
        peg_start_idx = id2index(peg_start_id)
        np.copyto(physics.data.qpos[peg_start_idx : peg_start_idx + 7], peg_pose)
        # print(f"randomized cube position to {cube_position}")

        socket_start_id = physics.model.name2id('blue_socket_joint', 'joint')
        socket_start_idx = id2index(socket_start_id)
        np.copyto(physics.data.qpos[socket_start_idx : socket_start_idx + 7], socket_pose)
        # print(f"randomized cube position to {cube_position}")

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

        touch_right_gripper = ("red_peg", "r_gripper_finger") in all_contact_pairs
        touch_left_gripper = ("socket-1", "l_gripper_finger") in all_contact_pairs or \
                             ("socket-2", "l_gripper_finger") in all_contact_pairs or \
                             ("socket-3", "l_gripper_finger") in all_contact_pairs or \
                             ("socket-4", "l_gripper_finger") in all_contact_pairs

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

class TransferCubeEETask(BimanualPiperEETask):
    def __init__(self, random=None, camera_names=None):
        super().__init__(random=random, camera_names=camera_names)
        self.max_reward = 4

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        self.initialize_robots(physics)
        # randomize box position
        cube_pose = sample_box_pose()
        box_start_idx = physics.model.name2id('red_box_joint', 'joint')
        np.copyto(physics.data.qpos[box_start_idx : box_start_idx + 7], cube_pose)
        # print(f"randomized cube position to {cube_position}")

        """
        for i in range(physics.model.njnt):
            joint_name = physics.model.joint(i).name
            qpos_start_index = physics.model.jnt_qposadr[i]
            if i < physics.model.njnt - 1:
                qpos_end_index = physics.model.jnt_qposadr[i+1]
                qpos_len = qpos_end_index - qpos_start_index
            else:
                qpos_len = physics.model.nq - qpos_start_index
            qpos_indices = list(range(qpos_start_index, qpos_start_index + qpos_len))
            print(f"qpos{qpos_indices} -> Joint '{joint_name}' (dof: {qpos_len})")
        for i in range(physics.model.nu):
            actuator_name = physics.model.actuator(i).name
            control_value = physics.data.ctrl[i]
            print(f"ctrl[{i}] -> Actuator '{actuator_name}': {control_value:.4f}")
        """
       
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

class MovingcubeEETask(BimanualPiperEETask):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 3

    def before_step(self, action, physics):
        action_left = action[:8]
        action_right = action[8:16]
        # left
        np.copyto(physics.data.mocap_pos[0], action_left[:3])
        np.copyto(physics.data.mocap_quat[0], action_left[3:7])
        # right
        np.copyto(physics.data.mocap_pos[1], action_right[:3])
        np.copyto(physics.data.mocap_quat[1], action_right[3:7])
        # gripper
        g_left_ctrl = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(action_left[7])
        g_right_ctrl = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(action_right[7])
        np.copyto(physics.data.ctrl[1:3], np.array([g_left_ctrl, g_right_ctrl]))

    

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        self.initialize_robots(physics)
        # randomize box position
        redcube_pose = sample_redbox_pose()
        greencube_pose = sample_greenbox_pose()
        bluecube_pose = sample_bluebox_pose()
        box_start_idx = physics.model.name2id('red_box_joint', 'joint')
        #socket_start_idx = physics.model.name2id('red_box_joint', 'joint')
        #stick_start_idx = physics.model.name2id('blue_box_joint', 'joint')
        np.copyto(physics.data.qpos[box_start_idx : box_start_idx + 7], redcube_pose)
        np.copyto(physics.data.qpos[box_start_idx + 7 : box_start_idx + 14], greencube_pose)
        np.copyto(physics.data.qpos[box_start_idx + 14 : box_start_idx + 21], bluecube_pose)

        physics.named.data.ctrl['belt_speed'] = BELT_MOVE_SPEED
        # print(f"randomized cube position to {cube_position}")

        """
        for i in range(physics.model.njnt):
            joint_name = physics.model.joint(i).name
            qpos_start_index = physics.model.jnt_qposadr[i]
            if i < physics.model.njnt - 1:
                qpos_end_index = physics.model.jnt_qposadr[i+1]
                qpos_len = qpos_end_index - qpos_start_index
            else:
                qpos_len = physics.model.nq - qpos_start_index
            qpos_indices = list(range(qpos_start_index, qpos_start_index + qpos_len))
            print(f"qpos{qpos_indices} -> Joint '{joint_name}' (dof: {qpos_len})")
        for i in range(physics.model.nu):
            actuator_name = physics.model.actuator(i).name
            control_value = physics.data.ctrl[i]
            print(f"ctrl[{i}] -> Actuator '{actuator_name}': {control_value:.4f}")
        """

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

class CoopEETask(BimanualPiperEETask):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 3

    def before_step(self, action, physics):
        action_left = action[:8]
        action_right = action[8:16]
        # left
        np.copyto(physics.data.mocap_pos[0], action_left[:3])
        np.copyto(physics.data.mocap_quat[0], action_left[3:7])
        # right
        np.copyto(physics.data.mocap_pos[1], action_right[:3])
        np.copyto(physics.data.mocap_quat[1], action_right[3:7])
        # gripper
        g_left_ctrl = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(action_left[7])
        g_right_ctrl = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(action_right[7])
        np.copyto(physics.data.ctrl[1:3], np.array([g_left_ctrl, g_right_ctrl]))

    

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        self.initialize_robots(physics)
        # randomize box position
        redcube_pose = sample_redbox_pose()
        greencube_pose = sample_greenbox_pose()
        bluecube_pose = sample_bluebox_pose()
        box_start_idx = physics.model.name2id('red_box_joint', 'joint')
        #socket_start_idx = physics.model.name2id('red_box_joint', 'joint')
        #stick_start_idx = physics.model.name2id('blue_box_joint', 'joint')
        np.copyto(physics.data.qpos[box_start_idx : box_start_idx + 7], redcube_pose)
        np.copyto(physics.data.qpos[box_start_idx + 7 : box_start_idx + 14], greencube_pose)
        np.copyto(physics.data.qpos[box_start_idx + 14 : box_start_idx + 21], bluecube_pose)

        physics.named.data.ctrl['belt_speed'] = BELT_MOVE_SPEED
        # print(f"randomized cube position to {cube_position}")

        """
        for i in range(physics.model.njnt):
            joint_name = physics.model.joint(i).name
            qpos_start_index = physics.model.jnt_qposadr[i]
            if i < physics.model.njnt - 1:
                qpos_end_index = physics.model.jnt_qposadr[i+1]
                qpos_len = qpos_end_index - qpos_start_index
            else:
                qpos_len = physics.model.nq - qpos_start_index
            qpos_indices = list(range(qpos_start_index, qpos_start_index + qpos_len))
            print(f"qpos{qpos_indices} -> Joint '{joint_name}' (dof: {qpos_len})")
        for i in range(physics.model.nu):
            actuator_name = physics.model.actuator(i).name
            control_value = physics.data.ctrl[i]
            print(f"ctrl[{i}] -> Actuator '{actuator_name}': {control_value:.4f}")
        """

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


class ManyCubesEETask(BimanualPiperEETask):
    def __init__(self, random=None, randomize_cube_colors=False, init_phase=1, camera_names=None):
        super().__init__(random=random, camera_names=camera_names)
        self.max_reward = 4
        self.randomize_cube_colors = randomize_cube_colors
        self.init_phase = init_phase

    def before_step(self, action, physics):
        action_left = action[:8]
        action_right = action[8:16]
        # left
        np.copyto(physics.data.mocap_pos[0], action_left[:3])
        np.copyto(physics.data.mocap_quat[0], action_left[3:7])
        # right
        np.copyto(physics.data.mocap_pos[1], action_right[:3])
        np.copyto(physics.data.mocap_quat[1], action_right[3:7])
        # gripper
        g_left_ctrl = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(action_left[7])
        g_right_ctrl = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(action_right[7])
        np.copyto(physics.data.ctrl[1:3], np.array([g_left_ctrl, g_right_ctrl]))

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        self.initialize_robots(physics)
        # randomize 10 cubes position
        # range mostly on the belt
        
        # Colors: R, G, B
        colors = [
            np.array([1, 0, 0, 1]), # R
            np.array([0, 1, 0, 1]), # G
            np.array([0, 0, 1, 1])  # B
        ]
        
        poses = {}
        
        if self.init_phase == 2:
            # Phase 2: G/B at goal, R at ORIGINAL
            # Green (8) at goal + noise
            noise_g = np.random.uniform(-0.02, 0.02, size=2)
            poses[8] = np.array([0 + noise_g[0], 0.1 + noise_g[1], 0.025, 1, 0, 0, 0])
            # Blue (9) at goal + noise
            noise_b = np.random.uniform(-0.02, 0.02, size=2)
            poses[9] = np.array([0.08 + noise_b[0], 0.1 + noise_b[1], 0.025, 1, 0, 0, 0])
            
            poses[7] = sample_redbox_pose()+np.array([0.2, 0, 0,0,0,0,0])
            ref_x = poses[7][0]
        else:
            poses[9] = sample_bluebox_pose()
            poses[8] = sample_greenbox_pose()
            poses[7] = sample_redbox_pose()
            
            # Fix: Anchor queue to the LEFT-MOST active cube (min X)
            xs = [poses[7][0], poses[8][0], poses[9][0]]
            ref_x = min(xs)
        
        queue_spacing = 0.22
        
        for i in range(10):
            if i in poses:
                cube_pose = poses[i]
            else:
                # i=6 -> 1 step behind 7
                step = 7 - i
                orig_x = ref_x - step * queue_spacing
                
                cube_x = orig_x + np.random.uniform(-0.01, 0.01)

                cube_y = np.random.uniform(0.30, 0.45) 
                cube_z = 0.02 
                
                cube_quat = np.array([1, 0, 0, 0])
                cube_pose = np.concatenate([[cube_x, cube_y, cube_z], cube_quat])
            
            joint_id = physics.model.name2id(f'cube_{i}_joint', 'joint')
            qpos_adr = physics.model.jnt_qposadr[joint_id]
            np.copyto(physics.data.qpos[qpos_adr : qpos_adr + 7], cube_pose)

            # Color logic
            if self.randomize_cube_colors:
                color = colors[np.random.randint(0, 3)]
            else:
                color_idx = (i + 2) % 3
                color = colors[color_idx]

            geom_id = physics.model.name2id(f'cube_{i}', 'geom')
            physics.model.geom_rgba[geom_id] = color

        physics.named.data.ctrl['belt_speed'] = BELT_MOVE_SPEED

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
