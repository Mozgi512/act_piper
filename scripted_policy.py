import numpy as np
import matplotlib.pyplot as plt
from pyquaternion import Quaternion

from piper_constants import SIM_TASK_CONFIGS,BELT_MOVE_SPEED
from piper_ee_sim_env import make_ee_sim_env

import IPython
e = IPython.embed


class BasePolicy:
    def __init__(self, inject_noise=False):
        self.inject_noise = inject_noise
        self.step_count = 0
        self.left_trajectory = None
        self.right_trajectory = None
        self.belt_trajectory = None

    def generate_trajectory(self, ts_first):
        raise NotImplementedError

    @staticmethod
    def interpolate(curr_waypoint, next_waypoint, t):
        t_frac = (t - curr_waypoint["t"]) / (next_waypoint["t"] - curr_waypoint["t"])
        curr_xyz = curr_waypoint['xyz']
        curr_quat = curr_waypoint['quat']
        curr_grip = curr_waypoint['gripper']
        next_xyz = next_waypoint['xyz']
        next_quat = next_waypoint['quat']
        next_grip = next_waypoint['gripper']
        xyz = curr_xyz + (next_xyz - curr_xyz) * t_frac
        quat = curr_quat + (next_quat - curr_quat) * t_frac
        gripper = curr_grip + (next_grip - curr_grip) * t_frac
        return xyz, quat, gripper
    
    @staticmethod
    def belt_interpolate(curr_waypoint, next_waypoint, t):
        t_frac = (t - curr_waypoint["t"]) / (next_waypoint["t"] - curr_waypoint["t"])
        curr_x = curr_waypoint['x']
        next_x = next_waypoint['x']
        x = curr_x + (next_x - curr_x) * t_frac
        return x
    
    def __call__(self, ts):
        # generate trajectory at first timestep, then open-loop execution
        if self.step_count == 0:
            self.generate_trajectory(ts)

        # obtain left and right waypoints
        if self.left_trajectory[0]['t'] == self.step_count:
            self.curr_left_waypoint = self.left_trajectory.pop(0)
        next_left_waypoint = self.left_trajectory[0]

        if self.right_trajectory[0]['t'] == self.step_count:
            self.curr_right_waypoint = self.right_trajectory.pop(0)
        next_right_waypoint = self.right_trajectory[0]

        # interpolate between waypoints to obtain current pose and gripper command
        left_xyz, left_quat, left_gripper = self.interpolate(self.curr_left_waypoint, next_left_waypoint, self.step_count)
        right_xyz, right_quat, right_gripper = self.interpolate(self.curr_right_waypoint, next_right_waypoint, self.step_count)

        # Inject noise
        if self.inject_noise:
            scale = 0.01
            left_xyz = left_xyz + np.random.uniform(-scale, scale, left_xyz.shape)
            right_xyz = right_xyz + np.random.uniform(-scale, scale, right_xyz.shape)

        action_left = np.concatenate([left_xyz, left_quat, [left_gripper]])
        action_right = np.concatenate([right_xyz, right_quat, [right_gripper]])

        self.step_count += 1
        return np.concatenate([action_left, action_right])

class PickAndTransferPolicy(BasePolicy):

    def generate_trajectory(self, ts_first):
        init_mocap_pose_right = ts_first.observation['mocap_pose_right']
        init_mocap_pose_left = ts_first.observation['mocap_pose_left']

        works_info = np.array(ts_first.observation['env_state'])
        box_xyz = works_info[:3]
        box_quat = works_info[3:7]


        # print(f"Generate trajectory for {box_xyz=}")

        gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
        gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[0.0, 1.0, 0.0], degrees=-60)

        meet_left_quat = Quaternion(axis=[1.0, 0.0, 0.0], degrees=90)

        meet_xyz = np.array([0, 0.25, 0.25])

        self.left_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 1}, # sleep
            {"t": 100, "xyz": meet_xyz + np.array([-0.1, 0, -0.0]), "quat": meet_left_quat.elements, "gripper": 1}, # approach meet position
            {"t": 260, "xyz": meet_xyz + np.array([0.02, 0, -0.0]), "quat": meet_left_quat.elements, "gripper": 1}, # move to meet position
            {"t": 310, "xyz": meet_xyz + np.array([0.02, 0, -0.0]), "quat": meet_left_quat.elements, "gripper": 0}, # close gripper
            {"t": 360, "xyz": meet_xyz + np.array([-0.1, 0, -0.0]), "quat": np.array([1, 0, 0, 0]), "gripper": 0}, # move left
            {"t": 400, "xyz": meet_xyz + np.array([-0.1, 0, -0.0]), "quat": np.array([1, 0, 0, 0]), "gripper": 0}, # stay
        ]

        self.right_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 1}, # sleep
            {"t": 90, "xyz": box_xyz + np.array([0, 0, 0.08]), "quat": gripper_pick_quat.elements, "gripper": 1}, # approach the cube
            {"t": 130, "xyz": box_xyz + np.array([0, 0, -0.01]), "quat": gripper_pick_quat.elements, "gripper": 1}, # go down
            {"t": 170, "xyz": box_xyz + np.array([0, 0, -0.01]), "quat": gripper_pick_quat.elements, "gripper": 0}, # close gripper
            {"t": 200, "xyz": meet_xyz + np.array([0.05, 0, 0]), "quat": gripper_pick_quat.elements, "gripper": 0}, # approach meet position
            {"t": 220, "xyz": meet_xyz, "quat": gripper_pick_quat.elements, "gripper": 0}, # move to meet position
            {"t": 310, "xyz": meet_xyz, "quat": gripper_pick_quat.elements, "gripper": 1}, # open gripper
            {"t": 360, "xyz": meet_xyz + np.array([0.1, 0, 0]), "quat": gripper_pick_quat.elements, "gripper": 1}, # move to right
            {"t": 400, "xyz": meet_xyz + np.array([0.1, 0, 0]), "quat": gripper_pick_quat.elements, "gripper": 1}, # stay
        ]

class PickMovingCubePolicy(BasePolicy):

    def generate_trajectory(self, ts_first):
        init_mocap_pose_right = ts_first.observation['mocap_pose_right']
        init_mocap_pose_left = ts_first.observation['mocap_pose_left']

        works_info = np.array(ts_first.observation['env_state'])
        redbox_xyz = works_info[0:3]
        redbox_quat = works_info[3:7]
        greenbox_xyz = works_info[7:10]
        greenbox_quat = works_info[10:14]
        bluebox_xyz = works_info[14:17]
        bluebox_quat = works_info[17:21]
        redbox_target_xyz = redbox_xyz + np.array([BELT_MOVE_SPEED*10, 0, 0])
        greenbox_target_xyz = greenbox_xyz + np.array([BELT_MOVE_SPEED*3+0.01, 0, 0])
        bluebox_target_xyz = bluebox_xyz + np.array([BELT_MOVE_SPEED*3+0.01, 0, 0])

        gripper_pick_quat_right = Quaternion(init_mocap_pose_right[3:])
        gripper_pick_quat_right = gripper_pick_quat_right * Quaternion(axis=[0.0, 1.0, 0.0], degrees=-30)
        gripper_assemble_quat_right = gripper_pick_quat_right * Quaternion(axis=[0.0, 1.0, 0.0], degrees=-90)


        gripper_pick_quat_left = Quaternion(init_mocap_pose_left[3:])
        gripper_pick_quat_left = gripper_pick_quat_left * Quaternion(axis=[0.0, 1.0, 0.0], degrees=30)
        gripper_assemble_quat_left = gripper_pick_quat_left * Quaternion(axis=[0.0, 1.0, 0.0], degrees=90)


        assemble_xyz = np.array([0, 0.3, 0.15])
        place_xyz = np.array([0, 0.1, 0.025])

        self.left_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 1}, # sleep
            {"t": 90, "xyz": greenbox_target_xyz + np.array([0, 0, 0.08]), "quat": gripper_pick_quat_left.elements, "gripper": 1}, # approach the cube
            {"t": 140, "xyz": greenbox_target_xyz + np.array([0, 0, 0.01]), "quat": gripper_pick_quat_left.elements, "gripper": 1}, # go down
            {"t": 170, "xyz": greenbox_target_xyz + np.array([0, 0, 0.01]), "quat": gripper_pick_quat_left.elements, "gripper": 0}, # close gripper
            {"t": 190, "xyz": greenbox_target_xyz + np.array([0, 0, 0.05]), "quat": gripper_pick_quat_left.elements, "gripper": 0}, # go up
            {"t": 280, "xyz": place_xyz + np.array([-0.05, 0, 0.05]), "quat": gripper_pick_quat_left.elements, "gripper": 0}, # approach place position
            {"t": 300, "xyz": place_xyz + np.array([-0.05, 0, 0]), "quat": gripper_pick_quat_left.elements, "gripper": 0}, # move to meet position
            {"t": 320, "xyz": place_xyz + np.array([-0.05, 0, 0]), "quat": gripper_pick_quat_left.elements, "gripper": 1}, # open gripper
            {"t": 340, "xyz": place_xyz + np.array([-0.05, 0, 0.05]), "quat": gripper_pick_quat_left.elements, "gripper": 1}, # exit
            {"t": 360, "xyz": place_xyz + np.array([-0.05, 0, 0.05]), "quat": gripper_pick_quat_left.elements, "gripper": 1}, # wait
            #{"t": 430, "xyz": redbox_target_xyz + np.array([0, 0, 0.08]), "quat": gripper_pick_quat_left.elements, "gripper": 1}, # approach the cube
            #{"t": 480, "xyz": redbox_target_xyz + np.array([0, 0, 0.01]), "quat": gripper_pick_quat_left.elements, "gripper": 1}, # go down
            #{"t": 510, "xyz": redbox_target_xyz + np.array([0, 0, 0.01]), "quat": gripper_pick_quat_left.elements, "gripper": 0}, # close gripper
            #{"t": 530, "xyz": redbox_target_xyz + np.array([0, 0, 0.05]), "quat": gripper_pick_quat_left.elements, "gripper": 0}, # go up
            #{"t": 620, "xyz": place_xyz + np.array([-0.05, 0, 0.05]), "quat": gripper_pick_quat_left.elements, "gripper": 0}, # approach place position
            #{"t": 640, "xyz": place_xyz + np.array([-0.05, 0, 0]), "quat": gripper_pick_quat_left.elements, "gripper": 0}, # move to meet position
            #{"t": 660, "xyz": place_xyz + np.array([-0.05, 0, 0]), "quat": gripper_pick_quat_left.elements, "gripper": 1}, # open gripper
        ]
        self.right_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 1}, # sleep
            {"t": 90, "xyz": bluebox_target_xyz + np.array([0, 0, 0.08]), "quat": gripper_pick_quat_right.elements, "gripper": 1}, # approach the cube
            {"t": 140, "xyz": bluebox_target_xyz + np.array([0, 0, 0.01]), "quat": gripper_pick_quat_right.elements, "gripper": 1}, # go down
            {"t": 170, "xyz": bluebox_target_xyz + np.array([0, 0, 0.01]), "quat": gripper_pick_quat_right.elements, "gripper": 0}, # close gripper
            {"t": 190, "xyz": bluebox_target_xyz + np.array([0, 0, 0.05]), "quat": gripper_pick_quat_right.elements, "gripper": 0}, # go up
            {"t": 280, "xyz": place_xyz + np.array([0.05, 0, 0.05]), "quat": gripper_pick_quat_right.elements, "gripper": 0}, # approach meet position
            {"t": 300, "xyz": place_xyz + np.array([0.05, 0, 0]), "quat": gripper_pick_quat_right.elements, "gripper": 0}, # move to meet position
            {"t": 320, "xyz": place_xyz + np.array([0.05, 0, 0]), "quat": gripper_pick_quat_right.elements, "gripper": 1},  # open gripper
            {"t": 340, "xyz": place_xyz + np.array([0.05, 0, 0.05]), "quat": gripper_pick_quat_right.elements, "gripper": 1}, # exit
            {"t": 360, "xyz": place_xyz + np.array([0.05, 0, 0.05]), "quat": gripper_pick_quat_right.elements, "gripper": 1}, # wait



            #{"t": 390, "xyz": place_xyz + np.array([-0.05, 0, 0.08]), "quat": gripper_pick_quat_right.elements, "gripper": 1}, # approach the cube
            #{"t": 440, "xyz": place_xyz + np.array([-0.05, 0, 0.01]), "quat": gripper_pick_quat_right.elements, "gripper": 1}, # go down
            #{"t": 470, "xyz": place_xyz + np.array([-0.05, 0, 0.01]), "quat": gripper_pick_quat_right.elements, "gripper": 0}, # close gripper
            #{"t": 490, "xyz": place_xyz + np.array([-0.05, 0, 0.05]), "quat": gripper_pick_quat_right.elements, "gripper": 0}, # go up
            #{"t": 520, "xyz": place_xyz + np.array([0.05, 0, 0.08]), "quat": gripper_pick_quat_right.elements, "gripper": 0}, # approach meet position
            #{"t": 540, "xyz": place_xyz + np.array([0.05, 0, 0.04]), "quat": gripper_pick_quat_right.elements, "gripper": 0}, # move to meet position
            #{"t": 560, "xyz": place_xyz + np.array([0.05, 0, 0.04]), "quat": gripper_pick_quat_right.elements, "gripper": 1},  # open gripper
            #{"t": 580, "xyz": place_xyz + np.array([0.05, 0, 0.1]), "quat": gripper_pick_quat_right.elements, "gripper": 1}, # exit
            #{"t": 660, "xyz": place_xyz + np.array([0.05, 0, 0.1]), "quat": gripper_pick_quat_right.elements, "gripper": 1}, # stay

        ]

class CoopPolicy(BasePolicy):

    def generate_trajectory(self, ts_first):
        init_mocap_pose_right = ts_first.observation['mocap_pose_right']
        init_mocap_pose_left = ts_first.observation['mocap_pose_left']

        works_info = np.array(ts_first.observation['env_state'])
        redbox_xyz = works_info[0:3]
        redbox_quat = works_info[3:7]
        greenbox_xyz = works_info[7:10]
        greenbox_quat = works_info[10:14]
        bluebox_xyz = works_info[14:17]
        bluebox_quat = works_info[17:21]
        redbox_target_xyz = redbox_xyz + np.array([BELT_MOVE_SPEED*7+0.02, 0, 0])
        redbox_coop_target_xyz = redbox_xyz + np.array([BELT_MOVE_SPEED*10, 0, 0])
        greenbox_target_xyz = greenbox_xyz + np.array([BELT_MOVE_SPEED*3+0.01, 0, 0])
        bluebox_target_xyz = bluebox_xyz + np.array([BELT_MOVE_SPEED*3+0.01, 0, 0])

        gripper_pick_quat_right = Quaternion(init_mocap_pose_right[3:])
        gripper_pick_quat_right = gripper_pick_quat_right * Quaternion(axis=[0.0, 1.0, 0.0], degrees=-30)
        gripper_assemble_quat_right = gripper_pick_quat_right * Quaternion(axis=[0.0, 1.0, 0.0], degrees=-90)


        gripper_pick_quat_left = Quaternion(init_mocap_pose_left[3:])
        gripper_pick_quat_left = gripper_pick_quat_left * Quaternion(axis=[0.0, 1.0, 0.0], degrees=30)
        gripper_assemble_quat_left = gripper_pick_quat_left * Quaternion(axis=[0.0, 1.0, 0.0], degrees=90)


        assemble_xyz = np.array([0.02, 0.25, 0.15])
        place_xyz = np.array([0, 0.1, 0.025])

        self.left_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 1}, # sleep
            {"t": 90, "xyz": greenbox_target_xyz + np.array([0, 0, 0.08]), "quat": gripper_pick_quat_left.elements, "gripper": 1}, # approach the cube
            {"t": 140, "xyz": greenbox_target_xyz + np.array([0, 0, 0.01]), "quat": gripper_pick_quat_left.elements, "gripper": 1}, # go down
            {"t": 170, "xyz": greenbox_target_xyz + np.array([0, 0, 0.01]), "quat": gripper_pick_quat_left.elements, "gripper": 0}, # close gripper
            {"t": 200, "xyz": assemble_xyz + np.array([-0.03, 0, 0.05]), "quat": gripper_pick_quat_left.elements, "gripper": 0}, # go up
            {"t": 220, "xyz": assemble_xyz + np.array([0, 0, 0.03]), "quat": gripper_pick_quat_left.elements, "gripper": 0}, # move to meet position
            {"t": 240, "xyz": assemble_xyz + np.array([0, 0, 0.03]), "quat": gripper_pick_quat_left.elements, "gripper": 1}, # open gripper
            {"t": 540, "xyz": assemble_xyz + np.array([0, 0, 0.03]), "quat": gripper_pick_quat_left.elements, "gripper": 1}, # open gripper
            #mix
            #{"t": 260, "xyz": assemble_xyz + np.array([-0.05, 0, 0.1]), "quat": gripper_pick_quat_left.elements, "gripper": 1}, # exit
            #{"t": 290, "xyz": redbox_target_xyz + np.array([0, 0, 0.08]), "quat": gripper_pick_quat_left.elements, "gripper": 1}, # approach the cube
            #{"t": 340, "xyz": redbox_target_xyz + np.array([0, 0, 0.01]), "quat": gripper_pick_quat_left.elements, "gripper": 1}, # go down
            #{"t": 370, "xyz": redbox_target_xyz + np.array([0, 0, 0.01]), "quat": gripper_pick_quat_left.elements, "gripper": 0}, # close gripper
            #{"t": 390, "xyz": redbox_target_xyz + np.array([0, 0, 0.05]), "quat": gripper_pick_quat_left.elements, "gripper": 0}, # go up
            #{"t": 480, "xyz": place_xyz + np.array([-0.05, 0, 0.05]), "quat": gripper_pick_quat_left.elements, "gripper": 0}, # approach place position
            #{"t": 500, "xyz": place_xyz + np.array([-0.05, 0, 0]), "quat": gripper_pick_quat_left.elements, "gripper": 0}, # move to meet position
            #{"t": 520, "xyz": place_xyz + np.array([-0.05, 0, 0]), "quat": gripper_pick_quat_left.elements, "gripper": 1}, # open gripper
            #{"t": 540, "xyz": place_xyz + np.array([-0.05, 0, 0.05]), "quat": gripper_pick_quat_left.elements, "gripper": 1}, # exit


            #coop
            #{"t": 260, "xyz": assemble_xyz + np.array([0, 0, 0.1]), "quat": gripper_pick_quat_left.elements, "gripper": 1}, # exit
            #{"t": 380, "xyz": assemble_xyz + np.array([0, 0, 0.1]), "quat": gripper_pick_quat_left.elements, "gripper": 1}, # stay
            #{"t": 440, "xyz": redbox_coop_target_xyz + np.array([0, 0, 0.08]), "quat": gripper_pick_quat_left.elements, "gripper": 1}, # approach the cube
            #{"t": 490, "xyz": redbox_coop_target_xyz + np.array([0, 0, 0.01]), "quat": gripper_pick_quat_left.elements, "gripper": 1}, # go down
            #{"t": 520, "xyz": redbox_coop_target_xyz + np.array([0, 0, 0.01]), "quat": gripper_pick_quat_left.elements, "gripper": 0}, # close gripper
            #{"t": 540, "xyz": redbox_coop_target_xyz + np.array([0, 0, 0.05]), "quat": gripper_pick_quat_left.elements, "gripper": 0}, # go up
            #{"t": 630, "xyz": place_xyz + np.array([-0.05, 0, 0.05]), "quat": gripper_pick_quat_left.elements, "gripper": 0}, # approach place position
            #{"t": 650, "xyz": place_xyz + np.array([-0.05, 0, 0]), "quat": gripper_pick_quat_left.elements, "gripper": 0}, # move to meet position
            #{"t": 670, "xyz": place_xyz + np.array([-0.05, 0, 0]), "quat": gripper_pick_quat_left.elements, "gripper": 1}, # open gripper
            #{"t": 690, "xyz": place_xyz + np.array([-0.05, 0, 0.05]), "quat": gripper_pick_quat_left.elements, "gripper": 1}, # exit


        ]
        self.right_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 1}, # sleep
            {"t": 90, "xyz": bluebox_target_xyz + np.array([0, 0, 0.08]), "quat": gripper_pick_quat_right.elements, "gripper": 1}, # approach the cube
            {"t": 140, "xyz": bluebox_target_xyz + np.array([0, 0, 0.015]), "quat": gripper_pick_quat_right.elements, "gripper": 1}, # go down
            {"t": 170, "xyz": bluebox_target_xyz + np.array([0, 0, 0.015]), "quat": gripper_pick_quat_right.elements, "gripper": 0}, # close gripper
            {"t": 200, "xyz": assemble_xyz + np.array([0, 0, -0.01]), "quat": gripper_pick_quat_right.elements, "gripper": 0}, # approach meet position
            {"t": 220, "xyz": assemble_xyz + np.array([0, 0, -0.01]), "quat": gripper_pick_quat_right.elements, "gripper": 0}, # move to meet position
            {"t": 250, "xyz": assemble_xyz + np.array([0, 0, -0.01]), "quat": gripper_pick_quat_right.elements, "gripper": 0},   #wait
            {"t": 340, "xyz": place_xyz + np.array([0.02, 0, 0.05]), "quat": gripper_pick_quat_right.elements, "gripper": 0}, # move to goal
            {"t": 360, "xyz": place_xyz + np.array([0.02, 0, 0]), "quat": gripper_pick_quat_right.elements, "gripper": 0}, # go down
            {"t": 380, "xyz": place_xyz + np.array([0.02, 0, 0]), "quat": gripper_pick_quat_right.elements, "gripper": 1}, # place
            {"t": 400, "xyz": place_xyz + np.array([0.02, 0, 0.1]), "quat": gripper_pick_quat_right.elements, "gripper": 1}, # stay
            {"t": 540, "xyz": place_xyz + np.array([0.02, 0, 0.1]), "quat": gripper_pick_quat_right.elements, "gripper": 1}, # stay

        ]


class InsertionPolicy(BasePolicy):

    def generate_trajectory(self, ts_first):
        init_mocap_pose_right = ts_first.observation['mocap_pose_right']
        init_mocap_pose_left = ts_first.observation['mocap_pose_left']

        peg_info = np.array(ts_first.observation['env_state'])[:7]
        peg_xyz = peg_info[:3]
        peg_quat = peg_info[3:]

        socket_info = np.array(ts_first.observation['env_state'])[7:]
        socket_xyz = socket_info[:3]
        socket_quat = socket_info[3:]

        gripper_pick_quat_right = Quaternion(init_mocap_pose_right[3:])
        gripper_pick_quat_right = gripper_pick_quat_right * Quaternion(axis=[0.0, 1.0, 0.0], degrees=-60)

        gripper_pick_quat_left = Quaternion(init_mocap_pose_right[3:])
        gripper_pick_quat_left = gripper_pick_quat_left * Quaternion(axis=[0.0, 1.0, 0.0], degrees=60)

        meet_xyz = np.array([0, 0.5, 0.15])
        lift_right = 0.00715

        self.left_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0}, # sleep
            {"t": 120, "xyz": socket_xyz + np.array([0, 0, 0.08]), "quat": gripper_pick_quat_left.elements, "gripper": 1}, # approach the cube
            {"t": 170, "xyz": socket_xyz + np.array([0, 0, -0.02]), "quat": gripper_pick_quat_left.elements, "gripper": 1}, # go down
            {"t": 220, "xyz": socket_xyz + np.array([0, 0, -0.02]), "quat": gripper_pick_quat_left.elements, "gripper": 0}, # close gripper
            {"t": 285, "xyz": meet_xyz + np.array([-0.1, 0, 0]), "quat": gripper_pick_quat_left.elements, "gripper": 0}, # approach meet position
            {"t": 340, "xyz": meet_xyz + np.array([-0.05, 0, 0]), "quat": gripper_pick_quat_left.elements,"gripper": 0},  # insertion
            {"t": 400, "xyz": meet_xyz + np.array([-0.05, 0, 0]), "quat": gripper_pick_quat_left.elements, "gripper": 0},  # insertion
        ]

        self.right_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
            {"t": 120, "xyz": peg_xyz + np.array([0, 0, 0.08]), "quat": gripper_pick_quat_right.elements, "gripper": 1}, # approach the cube
            {"t": 170, "xyz": peg_xyz + np.array([0, 0, -0.03]), "quat": gripper_pick_quat_right.elements, "gripper": 1}, # go down
            {"t": 220, "xyz": peg_xyz + np.array([0, 0, -0.03]), "quat": gripper_pick_quat_right.elements, "gripper": 0}, # close gripper
            {"t": 285, "xyz": meet_xyz + np.array([0.1, 0, lift_right]), "quat": gripper_pick_quat_right.elements, "gripper": 0}, # approach meet position
            {"t": 340, "xyz": meet_xyz + np.array([0.05, 0, lift_right]), "quat": gripper_pick_quat_right.elements, "gripper": 0},  # insertion
            {"t": 400, "xyz": meet_xyz + np.array([0.05, 0, lift_right]), "quat": gripper_pick_quat_right.elements, "gripper": 0},  # insertion

        ]


def test_policy(task_name):
    # example rolling out pick_and_transfer policy
    onscreen_render = True
    inject_noise = False

    # setup the environment
    episode_len = SIM_TASK_CONFIGS[task_name]['episode_len']
    if 'sim_transfer_cube' in task_name:
        env = make_ee_sim_env('sim_transfer_cube')
    elif 'sim_insertion' in task_name:
        env = make_ee_sim_env('sim_insertion')
    elif 'sim_moving_cube' in task_name:
        env = make_ee_sim_env('sim_moving_cube')
    elif 'sim_coop' in task_name:
        env = make_ee_sim_env('sim_coop') 
    else:
        raise NotImplementedError

    for episode_idx in range(2):
        ts = env.reset()
        episode = [ts]
        if onscreen_render:
            ax = plt.subplot()
            cam_image = env.physics.render(height=360, width=640, camera_id="top")
            plt_img = ax.imshow(cam_image)
            #plt_img = ax.imshow(ts.observation['images']['angle'])
            plt.ion()

        policy = CoopPolicy(inject_noise)
        for step in range(episode_len):
            action = policy(ts)
            ts = env.step(action)
            episode.append(ts)

            # === 接触判定デバッグ =======================================================
            # 現在のphysicsインスタンスを取得
            #physics = env.physics
            #print(f"--- Step {step}: {physics.data.ncon} contacts ---")
            #for i in range(physics.data.ncon):
            #    contact = physics.data.contact[i]
            #    geom1_name = physics.model.id2name(contact.geom1, 'geom')
            #    geom2_name = physics.model.id2name(contact.geom2, 'geom')
            #   print(f"  Contact {i}: {geom1_name} <--> {geom2_name}")

            # === 関節角度表示==========================================================
            physics = env.physics
            print(f"--- Step {step} Joint Angles ---")
            for i in range(physics.model.njnt):
                joint_name = physics.model.id2name(i, 'joint')
                if physics.model.joint(joint_name).type[0] != 0: # freejoint (type 0) を除外
                    qpos_index = physics.model.jnt_qposadr[i]
                    angle_rad = physics.data.qpos[qpos_index]
                    print(f"  {joint_name}: {angle_rad:.2f}")
            # ========================================================================
    
            if onscreen_render:
                cam_image = env.physics.render(height=360, width=640, camera_id="angle")
                plt_img.set_data(cam_image)
                #plt_img.set_data(ts.observation['images']['angle'])
                plt.pause(0.02)
        plt.close()

        episode_return = np.sum([ts.reward for ts in episode[1:]])
        if episode_return > 0:
            print(f"{episode_idx=} Successful, {episode_return=}")
        else:
            print(f"{episode_idx=} Failed")


if __name__ == '__main__':
    test_task_name = 'sim_coop_scripted'
    test_policy(test_task_name)