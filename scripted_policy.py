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
        self.success_t = None

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

class IndependentPolicy(BasePolicy):

    def generate_trajectory(self, ts_first):
        init_mocap_pose_right = ts_first.observation['mocap_pose_right']
        init_mocap_pose_left = ts_first.observation['mocap_pose_left']

        works_info = np.array(ts_first.observation['env_state'])
        # ManyCubesTask: 10 cubes. Red=7, Green=8, Blue=9.
        # Each cube 7 dims. 
        # Red (7): 7*7=49 -> [49:56]
        # Green (8): 8*7=56 -> [56:63]
        # Blue (9): 9*7=63 -> [63:70]
        
        redbox_xyz = works_info[49:52]
        redbox_quat = works_info[52:56]
        greenbox_xyz = works_info[56:59]
        greenbox_quat = works_info[59:63]
        bluebox_xyz = works_info[63:66]
        bluebox_quat = works_info[66:70]
        
        redbox_target_xyz = redbox_xyz + np.array([BELT_MOVE_SPEED*10-0.01, 0, 0])
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
            {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 1}, # initial pos
            {"t": 90, "xyz": greenbox_target_xyz + np.array([0, 0, 0.08]), "quat": gripper_pick_quat_left.elements, "gripper": 1}, # approach the cube
            {"t": 140, "xyz": greenbox_target_xyz + np.array([0, 0, 0.015]), "quat": gripper_pick_quat_left.elements, "gripper": 1}, # go down
            {"t": 170, "xyz": greenbox_target_xyz + np.array([0, 0, 0.015]), "quat": gripper_pick_quat_left.elements, "gripper": 0}, # close gripper
            {"t": 190, "xyz": greenbox_target_xyz + np.array([0, 0, 0.05]), "quat": gripper_pick_quat_left.elements, "gripper": 0}, # go up
            {"t": 260, "xyz": place_xyz + np.array([0, 0, 0.05]), "quat": gripper_pick_quat_left.elements, "gripper": 0}, # approach place position
            {"t": 280, "xyz": place_xyz + np.array([0, 0, 0]), "quat": gripper_pick_quat_left.elements, "gripper": 0}, # move to meet position
            {"t": 300, "xyz": place_xyz + np.array([0, 0, 0]), "quat": gripper_pick_quat_left.elements, "gripper": 1}, # open gripper
            {"t": 320, "xyz": place_xyz + np.array([0, 0, 0.05]), "quat": gripper_pick_quat_left.elements, "gripper": 1}, # exit
            {"t": 360, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 1},

            {"t": 380, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 1}, # initial pos
            {"t": 470, "xyz": redbox_target_xyz + np.array([0, 0, 0.08]), "quat": gripper_pick_quat_left.elements, "gripper": 1}, # approach cubic
            {"t": 520, "xyz": redbox_target_xyz + np.array([0, 0, 0.015]), "quat": gripper_pick_quat_left.elements, "gripper": 1}, # go down
            {"t": 550, "xyz": redbox_target_xyz + np.array([0, 0, 0.015]), "quat": gripper_pick_quat_left.elements, "gripper": 0}, # close gripper
            {"t": 570, "xyz": redbox_target_xyz + np.array([0, 0, 0.05]), "quat": gripper_pick_quat_left.elements, "gripper": 0}, # go up
            {"t": 640, "xyz": place_xyz + np.array([-0.08, 0, 0.05]), "quat": gripper_pick_quat_left.elements, "gripper": 0}, 
            {"t": 660, "xyz": place_xyz + np.array([-0.08, 0, 0]), "quat": gripper_pick_quat_left.elements, "gripper": 0}, 
            {"t": 680, "xyz": place_xyz + np.array([-0.08, 0, 0]), "quat": gripper_pick_quat_left.elements, "gripper": 1}, 
            {"t": 700, "xyz": place_xyz + np.array([-0.08, 0, 0.05]), "quat": gripper_pick_quat_left.elements, "gripper": 1}, # exit
            {"t": 740, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 1},
            #{"t": 360, "xyz": place_xyz + np.array([-0.05, 0, 0.05]), "quat": gripper_pick_quat_left.elements, "gripper": 1}, # wait
            #{"t": 430, "xyz": redbox_target_xyz + np.array([0, 0, 0.08]), "quat": gripper_pick_quat_left.elements, "gripper": 1}, # approach the cube
            #{"t": 480, "xyz": redbox_target_xyz + np.array([0, 0, 0.01]), "quat": gripper_pick_quat_left.elements, "gripper": 1}, # go down
            #{"t": 510, "xyz": redbox_target_xyz + np.array([0, 0, 0.01]), "quat": gripper_pick_quat_left.elements, "gripper": 0}, # close gripper
            #{"t": 530, "xyz": redbox_target_xyz + np.array([0, 0, 0.05]), "quat": gripper_pick_quat_left.elements, "gripper": 0}, # go up
            #{"t": 620, "xyz": place_xyz + np.array([-0.05, 0, 0.05]), "quat": gripper_pick_quat_left.elements, "gripper": 0}, # approach place position
            #{"t": 640, "xyz": place_xyz + np.array([-0.05, 0, 0]), "quat": gripper_pick_quat_left.elements, "gripper": 0}, # move to meet position
            #{"t": 660, "xyz": place_xyz + np.array([-0.05, 0, 0]), "quat": gripper_pick_quat_left.elements, "gripper": 1}, # open gripper
        ]
        self.right_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 1}, # initial pos
            {"t": 90, "xyz": bluebox_target_xyz + np.array([0, 0, 0.08]), "quat": gripper_pick_quat_right.elements, "gripper": 1}, # approach the cube
            {"t": 140, "xyz": bluebox_target_xyz + np.array([0, 0, 0.015]), "quat": gripper_pick_quat_right.elements, "gripper": 1}, # go down
            {"t": 170, "xyz": bluebox_target_xyz + np.array([0, 0, 0.015]), "quat": gripper_pick_quat_right.elements, "gripper": 0}, # close gripper
            {"t": 190, "xyz": bluebox_target_xyz + np.array([0, 0, 0.05]), "quat": gripper_pick_quat_right.elements, "gripper": 0}, # go up
            {"t": 260, "xyz": place_xyz + np.array([0.08, 0, 0.05]), "quat": gripper_pick_quat_right.elements, "gripper": 0}, # approach meet position
            {"t": 280, "xyz": place_xyz + np.array([0.08, 0, 0]), "quat": gripper_pick_quat_right.elements, "gripper": 0}, # move to meet position
            {"t": 300, "xyz": place_xyz + np.array([0.08, 0, 0]), "quat": gripper_pick_quat_right.elements, "gripper": 1},  # open gripper
            {"t": 320, "xyz": place_xyz + np.array([0.08, 0, 0.05]), "quat": gripper_pick_quat_right.elements, "gripper": 1}, # exit
            {"t": 360, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 1}, # wait
            {"t": 380, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 1}, 
            {"t": 440, "xyz": place_xyz + np.array([0, 0, 0.08]), "quat": gripper_pick_quat_right.elements, "gripper": 1}, 
            {"t": 490, "xyz": place_xyz + np.array([0, 0, 0.00]), "quat": gripper_pick_quat_right.elements, "gripper": 1}, 
            {"t": 520, "xyz": place_xyz + np.array([0, 0, 0.00]), "quat": gripper_pick_quat_right.elements, "gripper": 0}, 
            {"t": 540, "xyz": place_xyz + np.array([0, 0, 0.08]), "quat": gripper_pick_quat_right.elements, "gripper": 0}, 
            {"t": 580, "xyz": place_xyz + np.array([0.08, 0, 0.08]), "quat": gripper_pick_quat_right.elements, "gripper": 0},
            {"t": 600, "xyz": place_xyz + np.array([0.08, 0, 0.045]), "quat": gripper_pick_quat_right.elements, "gripper": 0}, 
            {"t": 620, "xyz": place_xyz + np.array([0.08, 0, 0.045]), "quat": gripper_pick_quat_right.elements, "gripper": 1}, 
            {"t": 640, "xyz": place_xyz + np.array([0.08, 0, 0.10]), "quat": gripper_pick_quat_right.elements, "gripper": 1}, 
            {"t": 680, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 1}, 
            {"t": 740, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 1}, 


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

        '''simple
        self.left_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 1}, # initial pos
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
            {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 1}, # initial pos
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
        '''

class IndependentPhase2Policy(BasePolicy):

    def generate_trajectory(self, ts_first):
        init_mocap_pose_right = ts_first.observation['mocap_pose_right']
        init_mocap_pose_left = ts_first.observation['mocap_pose_left']

        works_info = np.array(ts_first.observation['env_state'])
        # ManyCubesTask: 10 cubes. Red=7, Green=8, Blue=9.
        redbox_xyz = works_info[49:52]
        redbox_quat = works_info[52:56]
        greenbox_xyz = works_info[56:59]
        bluebox_xyz = works_info[63:66]
        
        # Red target: Moving at belt speed.
        # Starting from ORIGINAL position (approx -0.4).
        # To reach pickup zone (-0.1), needs ~0.3m -> 10s -> 500 steps?
        # Using the logic from IndependentPolicy (commented): BELT_MOVE_SPEED*10
        redbox_target_xyz = redbox_xyz + np.array([BELT_MOVE_SPEED*3+0.01, 0, 0])

        gripper_pick_quat_right = Quaternion(init_mocap_pose_right[3:])
        gripper_pick_quat_right = gripper_pick_quat_right * Quaternion(axis=[0.0, 1.0, 0.0], degrees=-30)

        gripper_pick_quat_left = Quaternion(init_mocap_pose_left[3:])
        gripper_pick_quat_left = gripper_pick_quat_left * Quaternion(axis=[0.0, 1.0, 0.0], degrees=30)
        
        place_xyz = np.array([0, 0.1, 0.025])
        
        self.left_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 1}, # initial pos
            {"t": 90, "xyz": redbox_target_xyz + np.array([0, 0, 0.08]), "quat": gripper_pick_quat_left.elements, "gripper": 1}, # approach cubic
            {"t": 140, "xyz": redbox_target_xyz + np.array([0, 0, 0.015]), "quat": gripper_pick_quat_left.elements, "gripper": 1}, # go down
            {"t": 170, "xyz": redbox_target_xyz + np.array([0, 0, 0.015]), "quat": gripper_pick_quat_left.elements, "gripper": 0}, # close gripper
            {"t": 190, "xyz": redbox_target_xyz + np.array([0, 0, 0.05]), "quat": gripper_pick_quat_left.elements, "gripper": 0}, # go up
            {"t": 260, "xyz": place_xyz + np.array([-0.08, 0, 0.05]), "quat": gripper_pick_quat_left.elements, "gripper": 0}, 
            {"t": 280, "xyz": place_xyz + np.array([-0.08, 0, 0]), "quat": gripper_pick_quat_left.elements, "gripper": 0}, 
            {"t": 300, "xyz": place_xyz + np.array([-0.08, 0, 0]), "quat": gripper_pick_quat_left.elements, "gripper": 1}, 
            {"t": 320, "xyz": place_xyz + np.array([-0.08, 0, 0.05]), "quat": gripper_pick_quat_left.elements, "gripper": 1}, # exit
            {"t": 360, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 1},


        ]
        
        self.right_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 1}, 
            # Pick Green at [-0.05]
            {"t": 60, "xyz": greenbox_xyz + np.array([0, 0, 0.08]), "quat": gripper_pick_quat_right.elements, "gripper": 1}, 
            {"t": 110, "xyz": greenbox_xyz + np.array([0, 0, 0.00]), "quat": gripper_pick_quat_right.elements, "gripper": 1}, 
            {"t": 140, "xyz": greenbox_xyz + np.array([0, 0, 0.00]), "quat": gripper_pick_quat_right.elements, "gripper": 0}, 
            {"t": 160, "xyz": greenbox_xyz + np.array([0, 0, 0.08]), "quat": gripper_pick_quat_right.elements, "gripper": 0}, 
            
            # Place on Blue at [+0.05] + height (0.045)
            {"t": 200, "xyz": bluebox_xyz + np.array([0, 0, 0.08]), "quat": gripper_pick_quat_right.elements, "gripper": 0},
            {"t": 220, "xyz": bluebox_xyz + np.array([0, 0, 0.045]), "quat": gripper_pick_quat_right.elements, "gripper": 0}, 
            {"t": 240, "xyz": bluebox_xyz + np.array([0, 0, 0.045]), "quat": gripper_pick_quat_right.elements, "gripper": 1}, 
            {"t": 260, "xyz": bluebox_xyz + np.array([0, 0, 0.10]), "quat": gripper_pick_quat_right.elements, "gripper": 1}, 
            {"t": 300, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 1}, 
            {"t": 360, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 1}, 
        ]

class CoopPolicy(BasePolicy):

    def generate_trajectory(self, ts_first):
        init_mocap_pose_right = ts_first.observation['mocap_pose_right']
        init_mocap_pose_left = ts_first.observation['mocap_pose_left']

        works_info = np.array(ts_first.observation['env_state'])
        redbox_xyz = works_info[49:52]
        redbox_quat = works_info[52:56]
        greenbox_xyz = works_info[56:59]
        greenbox_quat = works_info[59:63]
        bluebox_xyz = works_info[63:66]
        bluebox_quat = works_info[66:70]
        redbox_target_xyz = redbox_xyz + np.array([BELT_MOVE_SPEED*7+0.01, 0, 0])
        redbox_coop_target_xyz = redbox_xyz + np.array([BELT_MOVE_SPEED*10, 0, 0])
        greenbox_target_xyz = greenbox_xyz + np.array([BELT_MOVE_SPEED*3+0.01, 0, 0])
        bluebox_target_xyz = bluebox_xyz + np.array([BELT_MOVE_SPEED*3+0.01, 0, 0])

        gripper_pick_quat_right = Quaternion(init_mocap_pose_right[3:])
        gripper_pick_quat_right = gripper_pick_quat_right * Quaternion(axis=[0.0, 1.0, 0.0], degrees=-30)
        gripper_assemble_quat_right = gripper_pick_quat_right * Quaternion(axis=[0.0, 1.0, 0.0], degrees=-90)


        gripper_pick_quat_left = Quaternion(init_mocap_pose_left[3:])
        gripper_pick_quat_left = gripper_pick_quat_left * Quaternion(axis=[0.0, 1.0, 0.0], degrees=30)
        gripper_pick_higher_quat_left = gripper_pick_quat_left * Quaternion(axis=[0.0, 1.0, 0.0], degrees=30)


        assemble_xyz = np.array([0.02, 0.25, 0.1])
        place_xyz = np.array([0, 0.1, 0.025])

        self.left_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 1}, # sleep
            {"t": 90, "xyz": greenbox_target_xyz + np.array([0, 0, 0.08]), "quat": gripper_pick_quat_left.elements, "gripper": 1}, # approach the cube
            {"t": 140, "xyz": greenbox_target_xyz + np.array([0, 0, 0.02]), "quat": gripper_pick_quat_left.elements, "gripper": 1}, # go down
            {"t": 170, "xyz": greenbox_target_xyz + np.array([0, 0, 0.02]), "quat": gripper_pick_quat_left.elements, "gripper": 0}, # close gripper
            {"t": 210, "xyz": assemble_xyz + np.array([-0.03, 0, 0.05]), "quat": gripper_pick_quat_left.elements, "gripper": 0}, # go up
            {"t": 240, "xyz": assemble_xyz + np.array([0, 0, 0.03]), "quat": gripper_pick_quat_left.elements, "gripper": 0}, # move to meet position
            {"t": 260, "xyz": assemble_xyz + np.array([0, 0, 0.03]), "quat": gripper_pick_quat_left.elements, "gripper": 1}, # open gripper
            {"t": 280, "xyz": assemble_xyz + np.array([-0.1, 0, 0.05]), "quat": gripper_pick_quat_left.elements, "gripper": 1}, # exit


            {"t": 320, "xyz": redbox_target_xyz + np.array([0, 0, 0.08]), "quat": gripper_pick_quat_left.elements, "gripper": 1}, # approach the cube
            {"t": 360, "xyz": redbox_target_xyz + np.array([0, 0, 0.015]), "quat": gripper_pick_quat_left.elements, "gripper": 1}, # go down
            {"t": 380, "xyz": redbox_target_xyz + np.array([0, 0, 0.015]), "quat": gripper_pick_quat_left.elements, "gripper": 0}, # close gripper
            {"t": 400, "xyz": redbox_target_xyz + np.array([0, 0, 0.05]), "quat": gripper_pick_quat_left.elements, "gripper": 0}, # close gripper

            {"t": 460, "xyz": place_xyz + np.array([-0.05, 0, 0.05]), "quat": gripper_pick_quat_left.elements, "gripper": 0}, # go down
            {"t": 500, "xyz": place_xyz + np.array([-0.05, 0, 0]), "quat": gripper_pick_quat_left.elements, "gripper": 1}, # place
            {"t": 520, "xyz": place_xyz + np.array([-0.05, 0, 0.1]), "quat": gripper_pick_quat_left.elements, "gripper": 1}, # stay
            {"t": 560, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 1}, # return to start
            {"t": 580, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 1}, # stay until end

            #{"t": 320, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 1}, # return to start
            #{"t": 470, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 1}, # stay until end
          
        ]
        self.right_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 1}, # sleep
            {"t": 90, "xyz": bluebox_target_xyz + np.array([0, 0, 0.08]), "quat": gripper_pick_quat_right.elements, "gripper": 1}, # approach the cube
            {"t": 140, "xyz": bluebox_target_xyz + np.array([0, 0, 0.015]), "quat": gripper_pick_quat_right.elements, "gripper": 1}, # go down
            {"t": 170, "xyz": bluebox_target_xyz + np.array([0, 0, 0.015]), "quat": gripper_pick_quat_right.elements, "gripper": 0}, # close gripper
            {"t": 210, "xyz": assemble_xyz + np.array([0, 0, -0.01]), "quat": gripper_pick_quat_right.elements, "gripper": 0}, # approach meet position
            {"t": 240, "xyz": assemble_xyz + np.array([0, 0, -0.01]), "quat": gripper_pick_quat_right.elements, "gripper": 0}, # move to meet position
            {"t": 280, "xyz": assemble_xyz + np.array([0, 0, -0.01]), "quat": gripper_pick_quat_right.elements, "gripper": 0},   #wait
            {"t": 320, "xyz": place_xyz + np.array([0.02, 0, 0.05]), "quat": gripper_pick_quat_right.elements, "gripper": 0}, # move to goal
            {"t": 340, "xyz": place_xyz + np.array([0.02, 0, 0]), "quat": gripper_pick_quat_right.elements, "gripper": 0}, # go down
            {"t": 360, "xyz": place_xyz + np.array([0.02, 0, 0]), "quat": gripper_pick_quat_right.elements, "gripper": 1}, # place
            {"t": 380, "xyz": place_xyz + np.array([0.02, 0, 0.1]), "quat": gripper_pick_quat_right.elements, "gripper": 1}, # stay
            {"t": 430, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 1}, # return to start
            {"t": 580, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 1}, # stay until end
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



class VariableCoopPolicy(BasePolicy):
    def __init__(self, inject_noise=False):
        super().__init__(inject_noise)
        self.left_trajectory = []
        self.right_trajectory = []

    def generate_trajectory(self, ts_first):
        init_mocap_pose_right = ts_first.observation['mocap_pose_right']
        init_mocap_pose_left = ts_first.observation['mocap_pose_left']

        work_info = np.array(ts_first.observation['env_state'])
        # Extract cube positions (Red=7, Green=8, Blue=9) in ManyCubesTask
        red_xyz = work_info[49:52]
        green_xyz = work_info[56:59]
        blue_xyz = work_info[63:66]

        objects = [
            {'name': 'red', 'xyz': red_xyz, 'color': 'r'},
            {'name': 'green', 'xyz': green_xyz, 'color': 'g'},
            {'name': 'blue', 'xyz': blue_xyz, 'color': 'b'}
        ]
        # Sort by X coordinate (assuming smaller X is downstream/left)
        # Note: In Piper setup, belt moves from +X to -X ?
        # Need to verify direction. 
        # If +X (0.5) is upstream and -X (-0.5) is downstream.
        # Then Smaller X is Downstream (Left).
        objects.sort(key=lambda o: o['xyz'][0])
        
        obj_g = next(o for o in objects if o['name'] == 'green')
        obj_b = next(o for o in objects if o['name'] == 'blue')
        obj_r = next(o for o in objects if o['name'] == 'red')

        # Role Assignment
        # G vs B: Leftmost (Lower X) takes Left Arm, Rightmost (Higher X) takes Right Arm
        if obj_g['xyz'][0] < obj_b['xyz'][0]: 
            role_left = 'top' # G
            role_right = 'base' # B
        else: 
            role_left = 'base' # B
            role_right = 'top' # G
            
        task_left = []
        task_right = []
        r_index = objects.index(obj_r)
        
        # R Assignment Strategy:
        # 0 (Left): Left
        # 1 (Middle): Right (Changed from Left)
        # 2 (Right): Right
        if r_index == 0:
            r_arm = 'left'
            task_left.append('pick_r')
        else:
            r_arm = 'right'
            task_right.append('pick_r')

        # Role Assignment (New Logic):
        # The arm handling R should take 'Top' role (Place then Free)
        # The other arm takes 'Base' role (Wait then Transport)
        if r_arm == 'left':
            role_left = 'top'
            role_right = 'base'
        else:
            role_left = 'base'
            role_right = 'top'
            
        # Assign G/B tasks based on spatial position (Left arm takes Leftmost)
        # obj_g vs obj_b
        if obj_g['xyz'][0] < obj_b['xyz'][0]: # G is Left
            # Left takes G, Right takes B
            task_left.append(f'pick_g_{role_left}')
            task_right.append(f'pick_b_{role_right}')
        else: # B is Left
            # Left takes B, Right takes G
            task_left.append(f'pick_b_{role_left}')
            task_right.append(f'pick_g_{role_right}')
            
        # Initial Waypoints
        self.left_trajectory = [{"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 1}]
        self.right_trajectory = [{"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 1}]
        
        t_left = 0
        t_right = 0
        belt_speed = BELT_MOVE_SPEED
        meet_xyz = np.array([0, 0.3, 0.15]) 
        goal_xyz = np.array([0, 0.1, 0.025])
        
        
        def get_target_obj(task_name):
            if 'r' in task_name: return obj_r
            if 'g' in task_name: return obj_g
            if 'b' in task_name: return obj_b
            return None

        # Optimize execution order to avoid collisions (Pick objects closest to the other arm first)
        # Left Arm: Danger is Right side (+X). Prioritize picking Rightmost objects first.
        task_left.sort(key=lambda t: get_target_obj(t)['xyz'][0], reverse=True)
        
        # Right Arm: Danger is Left side (-X). Prioritize picking Leftmost objects first.
        task_right.sort(key=lambda t: get_target_obj(t)['xyz'][0], reverse=False)
        
        # Priority Update: G/B > R
        # Force 'pick_r' to the end of the list if present
        if 'pick_r' in task_left:
            task_left.remove('pick_r')
            task_left.append('pick_r')
            
        if 'pick_r' in task_right:
            task_right.remove('pick_r')
            task_right.append('pick_r')
        
        sync_time_top = None
        sync_time_base = None
        
        # Queue-based Scheduler
        # Sort tasks by priority (R vs G/B determined by Geometric Sort)
        # Process tasks: If R, do immediately. If G/B, wait for both arms to be ready then sync.
        
        while len(task_left) > 0 or len(task_right) > 0:
            next_l = task_left[0] if len(task_left) > 0 else None
            next_r = task_right[0] if len(task_right) > 0 else None
            
            # Case 1: Independent R (Priority) on Left
            if next_l == 'pick_r':
                t_left = self.add_pick_place(self.left_trajectory, t_left, red_xyz, goal_xyz, belt_speed, is_left=True, offset_x=-0.08)
                task_left.pop(0)
                continue
                
            # Case 2: Independent R (Priority) on Right
            if next_r == 'pick_r':
                t_right = self.add_pick_place(self.right_trajectory, t_right, red_xyz, goal_xyz, belt_speed, is_left=False, offset_x=0.08)
                task_right.pop(0)
                continue
                
            # Case 3: Sync Task (G/B) - Both must be ready
            if (next_l and 'pick' in next_l) and (next_r and 'pick' in next_r):
                # Sync Start
                t_joint_start = max(t_left, t_right)
                
                # Update both to start together
                if t_left < t_joint_start:
                    self.add_wait(self.left_trajectory, t_joint_start)
                    t_left = t_joint_start
                if t_right < t_joint_start:
                    self.add_wait(self.right_trajectory, t_joint_start)
                    t_right = t_joint_start
                    
                # 1. Execute Approachs (Pick -> Meet)
                # We need to know who is Top and who is Base to determine duration/logic
                # But my helpers add_pick_top/base do the move.
                # Let's execute them and get arrival times.
                
                t_arrival_left = 0
                t_arrival_right = 0
                
                # Left Arm Action
                if 'top' in next_l:
                    is_left_top = True
                    target = green_xyz if 'g' in next_l else blue_xyz
                    t_arrival_left = self.add_pick_top(self.left_trajectory, t_left, target, meet_xyz, belt_speed, is_left=True)
                else: # Base
                    is_left_top = False
                    target = green_xyz if 'g' in next_l else blue_xyz
                    t_arrival_left = self.add_pick_base(self.left_trajectory, t_left, target, meet_xyz, goal_xyz, belt_speed, is_left=True)

                # Right Arm Action
                if 'top' in next_r:
                    target = green_xyz if 'g' in next_r else blue_xyz
                    t_arrival_right = self.add_pick_top(self.right_trajectory, t_right, target, meet_xyz, belt_speed, is_left=False)
                else:
                    target = green_xyz if 'g' in next_r else blue_xyz
                    t_arrival_right = self.add_pick_base(self.right_trajectory, t_right, target, meet_xyz, goal_xyz, belt_speed, is_left=False)
                
                # 2. Synchronize Assembly at Meet Point
                t_meet = max(t_arrival_left, t_arrival_right)
                
                # Pad to t_meet
                if t_arrival_left < t_meet:
                    self.add_wait(self.left_trajectory, t_meet)
                if t_arrival_right < t_meet:
                    self.add_wait(self.right_trajectory, t_meet)
                    
                # 3. Execute Assembly & Transport
                # Top Arm: Place -> Exit
                # Base Arm: Wait -> Transport
                
                if is_left_top:
                    # Left is Top, Right is Base
                    t_fin_top = self.add_assembly_top_action(self.left_trajectory, t_meet, meet_xyz, is_left=True)
                    t_left = t_fin_top
                    
                    # Right (Base) waits for assembly then transports
                    assemble_duration = t_fin_top - t_meet
                    # Add wait for assembly
                    t_transport_start = t_meet + assemble_duration
                    self.add_wait(self.right_trajectory, t_transport_start)
                    # Transport
                    t_right = self.add_base_transport(self.right_trajectory, t_transport_start, goal_xyz, is_left=False, offset_x=0.02)
                    
                else:
                    # Right is Top, Left is Base
                    t_fin_top = self.add_assembly_top_action(self.right_trajectory, t_meet, meet_xyz, is_left=False)
                    t_right = t_fin_top
                    
                    assemble_duration = t_fin_top - t_meet
                    t_transport_start = t_meet + assemble_duration
                    self.add_wait(self.left_trajectory, t_transport_start)
                    t_left = self.add_base_transport(self.left_trajectory, t_transport_start, goal_xyz, is_left=True, offset_x=-0.02)

                task_left.pop(0)
                task_right.pop(0)
                continue
                
            break

        # Final Synchronization: Pad shorter trajectory
        max_t = max(t_left, t_right) + 50 # Add buffer at end
        self.success_t = max_t
        
        last_left = self.left_trajectory[-1]
        if last_left['t'] < max_t:
            self.left_trajectory.append({"t": max_t, "xyz": last_left['xyz'], "quat": last_left['quat'], "gripper": last_left['gripper']})
            
        last_right = self.right_trajectory[-1]
        if last_right['t'] < max_t:
            self.right_trajectory.append({"t": max_t, "xyz": last_right['xyz'], "quat": last_right['quat'], "gripper": last_right['gripper']})

        # Add hold-forever waypoint to prevent index error if simulation runs long
        final_t = 10000 # Large enough to cover max episode steps
        final_left = self.left_trajectory[-1]
        self.left_trajectory.append({"t": final_t, "xyz": final_left['xyz'], "quat": final_left['quat'], "gripper": final_left['gripper']})
        
        final_right = self.right_trajectory[-1]
        self.right_trajectory.append({"t": final_t, "xyz": final_right['xyz'], "quat": final_right['quat'], "gripper": final_right['gripper']})

    def add_wait(self, traj, until_t):
        if not traj: return
        last = traj[-1]
        if last['t'] < until_t:
            traj.append({"t": until_t, "xyz": last['xyz'], "quat": last['quat'], "gripper": last['gripper']})

    def get_quat(self, is_left, mode='pick', top_down=False):
        if top_down:
            # Vertical pick (Downwards)
            # User reported 90 degrees was 180 deg reversed. Flipping to -90.
            q = Quaternion(axis=[0, 1, 0], degrees=-90)
            return q.elements

        if is_left:
            q = Quaternion(axis=[0, 1, 0], degrees=30)
            if mode == 'assemble': 
                pass
            return q.elements
        else:
            q = Quaternion(axis=[0, 1, 0], degrees=-30)
            if mode == 'assemble':
                pass
            return q.elements

    def add_pick_place(self, traj, start_t, obj_xyz, goal_xyz, belt_speed, is_left, offset_x=0.0):
        # Move to Object -> Pick -> Move to Goal -> Place -> Return
        approach_dur = 80
        intercept_t = start_t + approach_dur
        target_xyz = obj_xyz + np.array([belt_speed * (intercept_t * 0.02) + 0.02, 0, 0])
        
        # Adaptive Orientation: If Right Arm and object is far right (>0.25), use Top-Down
        top_down = False
        if not is_left and target_xyz[0] > 0.2:
            top_down = True
            
        q_pick = self.get_quat(is_left, 'pick', top_down=top_down)
        
        traj.append({"t": intercept_t, "xyz": target_xyz + [0, 0, 0.08], "quat": q_pick, "gripper": 1})
        traj.append({"t": intercept_t + 20, "xyz": target_xyz + [0, 0, 0], "quat": q_pick, "gripper": 1})
        traj.append({"t": intercept_t + 40, "xyz": target_xyz + [0, 0, 0], "quat": q_pick, "gripper": 0})
        traj.append({"t": intercept_t + 60, "xyz": target_xyz + [0, 0, 0.08], "quat": q_pick, "gripper": 0})
        
        current_t = intercept_t + 60
        place_pos = goal_xyz + [offset_x, 0, 0.05]
        traj.append({"t": current_t + 60, "xyz": place_pos, "quat": q_pick, "gripper": 0})
        traj.append({"t": current_t + 80, "xyz": place_pos - [0, 0, 0.05], "quat": q_pick, "gripper": 0})
        traj.append({"t": current_t + 100, "xyz": place_pos - [0, 0, 0.05], "quat": q_pick, "gripper": 1})
        traj.append({"t": current_t + 120, "xyz": place_pos, "quat": q_pick, "gripper": 1})
        
        return current_t + 120

    def add_pick_top(self, traj, start_t, obj_xyz, meet_xyz, belt_speed, is_left):
        approach_dur = 80
        intercept_t = start_t + approach_dur
        target_xyz = obj_xyz + np.array([belt_speed * (intercept_t * 0.02) + 0.02, 0, 0])
        q_pick = self.get_quat(is_left, 'pick')
        
        traj.append({"t": intercept_t, "xyz": target_xyz + [0, 0, 0.08], "quat": q_pick, "gripper": 1})
        traj.append({"t": intercept_t + 20, "xyz": target_xyz + [0, 0, 0], "quat": q_pick, "gripper": 1})
        traj.append({"t": intercept_t + 40, "xyz": target_xyz + [0, 0, 0], "quat": q_pick, "gripper": 0})
        traj.append({"t": intercept_t + 60, "xyz": target_xyz + [0, 0, 0.08], "quat": q_pick, "gripper": 0})
        
        current_t = intercept_t + 60
        # Move to Meet (+Z offset for Top)
        traj.append({"t": current_t + 60, "xyz": meet_xyz + [0, 0, 0.05], "quat": q_pick, "gripper": 0})
        
        return current_t + 60

    def add_assembly_top_action(self, traj, start_t, meet_xyz, is_left):
        q_pick = self.get_quat(is_left, 'pick')
        # Lower -> Open -> Raise
        traj.append({"t": start_t + 20, "xyz": meet_xyz + [0, 0, 0.025], "quat": q_pick, "gripper": 0}) # Touch
        traj.append({"t": start_t + 40, "xyz": meet_xyz + [0, 0, 0.025], "quat": q_pick, "gripper": 1}) # Release
        traj.append({"t": start_t + 60, "xyz": meet_xyz + [0, 0, 0.10], "quat": q_pick, "gripper": 1}) # Exit
        return start_t + 60

    def add_pick_base(self, traj, start_t, obj_xyz, meet_xyz, goal_xyz, belt_speed, is_left):
        approach_dur = 80
        intercept_t = start_t + approach_dur
        target_xyz = obj_xyz + np.array([belt_speed * (intercept_t * 0.02) + 0.02, 0, 0])
        q_pick = self.get_quat(is_left, 'pick')
        
        traj.append({"t": intercept_t, "xyz": target_xyz + [0, 0, 0.08], "quat": q_pick, "gripper": 1})
        traj.append({"t": intercept_t + 20, "xyz": target_xyz + [0, 0, 0], "quat": q_pick, "gripper": 1})
        traj.append({"t": intercept_t + 40, "xyz": target_xyz + [0, 0, 0], "quat": q_pick, "gripper": 0})
        traj.append({"t": intercept_t + 60, "xyz": target_xyz + [0, 0, 0.08], "quat": q_pick, "gripper": 0})
        
        current_t = intercept_t + 60
        # Move to Meet (Base pos)
        traj.append({"t": current_t + 60, "xyz": meet_xyz - [0, 0, 0.025], "quat": q_pick, "gripper": 0})
        
        return current_t + 60

    def add_base_transport(self, traj, start_t, goal_xyz, is_left, offset_x=0.0):
        q_pick = self.get_quat(is_left, 'pick')
        place_pos = goal_xyz + [offset_x, 0, 0.0]
        traj.append({"t": start_t + 60, "xyz": place_pos + [0, 0, 0.05], "quat": q_pick, "gripper": 0})
        traj.append({"t": start_t + 80, "xyz": place_pos, "quat": q_pick, "gripper": 0})
        traj.append({"t": start_t + 100, "xyz": place_pos, "quat": q_pick, "gripper": 1})
        traj.append({"t": start_t + 120, "xyz": place_pos + [0, 0, 0.1], "quat": q_pick, "gripper": 1})
        return start_t + 120

def test_policy(task_name):
    # example rolling out pick_and_transfer policy
    onscreen_render = True
    inject_noise = False

    # setup the environment
    if task_name == 'sim_variable_coop_scripted':
        episode_len = 1000 # custom len
        env = make_ee_sim_env('sim_coop') # Re-use coop env but with variable policy if we inject random pose
        # For testing variable coop: we need random start positions?
        # env = make_ee_sim_env('sim_independent_phase2') # Phase2 has random locs but maybe too constrained?
        # Let's use sim_coop env but we might need to manually override box positions in test_policy loop to verify robustness
    elif task_name in SIM_TASK_CONFIGS:
        episode_len = SIM_TASK_CONFIGS[task_name]['episode_len']
        if 'sim_transfer_cube' in task_name:
            env = make_ee_sim_env('sim_transfer_cube')
        elif 'sim_insertion' in task_name:
            env = make_ee_sim_env('sim_insertion')
        elif 'sim_moving_cube' in task_name:
            env = make_ee_sim_env('sim_moving_cube')
        elif 'sim_independent_phase2' in task_name:
            env = make_ee_sim_env(task_name)
        elif 'sim_coop' in task_name:
            env = make_ee_sim_env('sim_coop')
        else:
            raise NotImplementedError
    else:
        # Fallback
        episode_len = 1000
        env = make_ee_sim_env('sim_coop')

    for episode_idx in range(5): # Run 5 times to test different orders if random
        ts = env.reset()
        
        # Override positions for Variable Coop Test to ensure randomness
        if task_name == 'sim_variable_coop_scripted':
            # Manually inject random R, G, B positions on belt
            # Range: 0.2 to -0.4?
            # Belt moves X+ -> X-?
            # Let's generate 3 random X in [-0.1, 0.4] (safe range)
            positions = np.random.uniform(-0.1, 0.4, 3) 
            # Ensure spacing
            # Simple manual assignment for test
            # idx 7,8,9 = R,G,B
            # Need to set qpos
            # R=7 (offset 16+1 (belt) +1(ext) + 7*7 = 18+49?)
            # piper_ee_sim_env.ManyCubesEETask sets positions in initialize_episode.
            # We can't easily override mapping here without accessing physics directly.
            # But we can assume the Env already randomized them if we used a randomized env.
            # CoopTask usually has fixed positions?
            # Let's trust make_ee_sim_env('sim_coop') for now, but maybe it only generates one pattern?
            # Yes, CoopEETask generates fixed.
            # We should use ManyCubesEETask with init_phase=2 or similar if accessible.
            # Or manually set qpos here.
            physics = env.physics
            # Get qpos addresses for cubes
            # R=cube_7, G=cube_8, B=cube_9
            # X coordinates
            # Let's shuffle the positions of R, G, B
            # Standard positions in Coop: R=[-0.19], G=[-0.05], B=[0.09] (approx) relative.
            # Let's define 3 slots and shuffle assignment
            slots = [0.2, 0.0, -0.2]
            np.random.shuffle(slots)
            
            # cube_7 (R)
            start_idx_r = physics.model.jnt_qposadr[physics.model.name2id('cube_7_joint', 'joint')]
            physics.data.qpos[start_idx_r] = slots[0]
            physics.data.qpos[start_idx_r+1] = 0.35 # Y
            
            # cube_8 (G)
            start_idx_g = physics.model.jnt_qposadr[physics.model.name2id('cube_8_joint', 'joint')]
            physics.data.qpos[start_idx_g] = slots[1]
            physics.data.qpos[start_idx_g+1] = 0.35
            
            # cube_9 (B)
            start_idx_b = physics.model.jnt_qposadr[physics.model.name2id('cube_9_joint', 'joint')]
            physics.data.qpos[start_idx_b] = slots[2]
            physics.data.qpos[start_idx_b+1] = 0.35
            
            # Update env state?
            ts = env._task.get_observation(physics) # Refresh obs
            # But ts structure might need manual update if we are not careful.
            # calling env.step with no-op might be better but we haven't started policy.
            # Hack: Manually construct TS-like obj or just pass physics to generation?
            # Policy uses ts.observation['env_state'].
            # get_observation re-reads from physics, so we are good if we call it.
            # However `env.reset()` returns a TimeStep. we need to update it.
            pass

        episode = [ts]
        if onscreen_render:
            ax = plt.subplot()
            cam_image = env.physics.render(height=360, width=640, camera_id="top")
            plt_img = ax.imshow(cam_image)
            #plt_img = ax.imshow(ts.observation['images']['angle'])
            plt.ion()

        if 'sim_transfer_cube' in task_name:
            policy = PickAndTransferPolicy(inject_noise)
        elif 'sim_insertion' in task_name:
            policy = InsertionPolicy(inject_noise)
        elif 'sim_independent_phase2' in task_name:
            policy = IndependentPhase2Policy(inject_noise)
        elif 'sim_moving_cube' in task_name or 'sim_independent' in task_name:
            policy = IndependentPolicy(inject_noise)
        elif 'sim_variable_coop_scripted' in task_name:
            # Need to refresh TS because we modified qpos
            obs = env._task.get_observation(env.physics)
            from dm_env import TimeStep, StepType
            ts = TimeStep(step_type=StepType.FIRST, reward=0, discount=1, observation=obs)
            policy = VariableCoopPolicy(inject_noise)
        elif 'sim_coop' in task_name:
            policy = CoopPolicy(inject_noise)
        else:
            raise NotImplementedError

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
            # physics = env.physics
            # print(f"--- Step {step} Joint Angles ---")
            # for i in range(physics.model.njnt):
            #     joint_name = physics.model.id2name(i, 'joint')
            #     if physics.model.joint(joint_name).type[0] != 0: # freejoint (type 0) を除外
            #         qpos_index = physics.model.jnt_qposadr[i]
            #         angle_rad = physics.data.qpos[qpos_index]
            #         print(f"  {joint_name}: {angle_rad:.2f}")
            # ========================================================================
    
            if onscreen_render:
                cam_image = env.physics.render(height=360, width=640, camera_id="top")
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
    test_policy('sim_variable_coop_scripted')