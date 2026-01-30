import pathlib

### Task parameters
DATA_DIR = '/home/act/act_piper/piper_scripted_dataset'
SIM_TASK_CONFIGS = {
    'sim_transfer_cube_scripted':{
        'dataset_dir': DATA_DIR + '/sim_transfer_cube_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['top']
    },

    'sim_transfer_cube_human':{
        'dataset_dir': DATA_DIR + '/sim_transfer_cube_human',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['top']
    },
    'sim_moving_cube_scripted':{
        #'dataset_dir': DATA_DIR + '/sim_moving_cube_scripted',
        'dataset_dir': DATA_DIR + '/independent',
        'num_episodes': 50,
        'episode_len': 360,
        'camera_names': ['top']
    },
    'sim_coop_scripted':{
        'dataset_dir': DATA_DIR + '/cooperation',
        'num_episodes': 100,
        'episode_len': 580,
        'camera_names': ['top']
    },
    'sim_independent_scripted':{
        #'dataset_dir': DATA_DIR + '/sim_moving_cube_scripted',
        'dataset_dir': DATA_DIR + '/independent_full',
        'num_episodes': 100,
        'episode_len': 740,
        'camera_names': ['top']
    },
    'sim_independent_phase2_scripted':{
        'dataset_dir': DATA_DIR + '/sim_independent_2phases',
        'num_episodes': 200,
        'episode_len': 360,
        'camera_names': ['top']
    },
    'sim_four_objects_scripted':{
        'dataset_dir': DATA_DIR + '/sim_four_objects',
        'num_episodes': 100,
        'episode_len': 700,
        'camera_names': ['top']
    },
    'sim_many_cubes':{
        'dataset_dir': DATA_DIR + '/sim_many_cubes',
        'num_episodes': 80,
        'episode_len': 1430,
        'camera_names': ['top']
    },
    'sim_coop_phase1_scripted':{
        'dataset_dir': DATA_DIR + '/cooperation_phase1',
        'num_episodes': 100,
        'episode_len': 280,
        'camera_names': ['top']
    },
    'sim_coop_phase2_left_scripted':{
        'dataset_dir': DATA_DIR + '/cooperation_phase2_left',
        'num_episodes': 100,
        'episode_len': 400,
        'camera_names': ['top']
    },
    'sim_coop_phase2_right_scripted':{
        'dataset_dir': DATA_DIR + '/cooperation_phase2_right',
        'num_episodes': 100,
        'episode_len': 400,
        'camera_names': ['top']
    },
    'sim_four_objects_scripted':{
        'dataset_dir': DATA_DIR + '/four_objects',
        'num_episodes': 100,
        'episode_len': 680,
        'camera_names': ['top']
    },
    'sim_dataset_i': {
        'dataset_dir': DATA_DIR + '/sim_dataset_i',
        'num_episodes': 100,
        'episode_len': 400,
        'camera_names': ['top']
    },
    'sim_dataset_c': {
        'dataset_dir': DATA_DIR + '/sim_dataset_c',
        'num_episodes': 100,
        'episode_len': 520,
        'camera_names': ['top']
    },
}

### Simulation envs fixed constants
DT = 0.02
BELT_MOVE_SPEED = 0.035 #m/s
JOINT_NAMES = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
#START_ARM_POSE = [2.2, 1.1, -0.5, 1.9, -2.1, -0.8, 0.02, -0.02, -2.2, 1.1, -0.5, -1.9, -2.1, 1, 0.02, -0.02]
#START_ARM_POSE = [1.80,1.60,-0.90, 1.63,-1.79,-0.87,0.03,-0.03,-1.78,1.57,-0.90,-1.68,-1.73,0.99,0.03,-0.03]
#START_ARM_POSE = [1.93,1.45,-1.08, 1.70,-1.87,-1.18,0.03,-0.03,-1.93,1.45,-1.08,-1.70,-1.87,1.18,0.03,-0.03]
#START_ARM_POSE = [0,0.36,-0.14, 0,-0.27,0,0.03,-0.03,0,0.36,-0.14,0,-0.27,0,0.03,-0.03]
#START_ARM_POSE = [0.76,1.23,-0.42,0.92,-1.05,-0.58,0.03,-0.03,-0.76,1.23,-0.42,-0.92,-1.05,0.58,0.03,-0.03]
#START_ARM_POSE = [1.01,1.40,-1.04,1.95,-0.91,-1.49,0.03,-0.04,-1.00,1.39,-1.05,-1.97,-0.91,1.53,0.03,-0.04]
START_ARM_POSE = [0.83, 1.62, -0.90, 1.56, -0.69, -1.05, 0.03, -0.04, -0.83, 1.62, -0.90, -1.57, -0.68, 1.07, 0.03, -0.04]


XML_DIR = str(pathlib.Path(__file__).parent.resolve()) + '/mujoco_piper/' # note: absolute path

# Left finger position limits (qpos[7]), right_finger = -1 * left_finger
MASTER_GRIPPER_POSITION_OPEN = 0.035
MASTER_GRIPPER_POSITION_CLOSE = 0.005
PUPPET_GRIPPER_POSITION_OPEN = 0.035
PUPPET_GRIPPER_POSITION_CLOSE = 0.005

# Gripper joint limits (qpos[6])
#need to be fixed using real robot piper
MASTER_GRIPPER_JOINT_OPEN = 0.035
MASTER_GRIPPER_JOINT_CLOSE = 0.005
PUPPET_GRIPPER_JOINT_OPEN = 0.035
PUPPET_GRIPPER_JOINT_CLOSE = 0.005

############################ Helper functions ############################

MASTER_GRIPPER_POSITION_NORMALIZE_FN = lambda x: (x - MASTER_GRIPPER_POSITION_CLOSE) / (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE)
PUPPET_GRIPPER_POSITION_NORMALIZE_FN = lambda x: (x - PUPPET_GRIPPER_POSITION_CLOSE) / (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE)
MASTER_GRIPPER_POSITION_UNNORMALIZE_FN = lambda x: x * (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE) + MASTER_GRIPPER_POSITION_CLOSE
PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN = lambda x: x * (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE) + PUPPET_GRIPPER_POSITION_CLOSE
MASTER2PUPPET_POSITION_FN = lambda x: PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(MASTER_GRIPPER_POSITION_NORMALIZE_FN(x))

MASTER_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - MASTER_GRIPPER_JOINT_CLOSE) / (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE)
PUPPET_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - PUPPET_GRIPPER_JOINT_CLOSE) / (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE)
MASTER_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE) + MASTER_GRIPPER_JOINT_CLOSE
PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE) + PUPPET_GRIPPER_JOINT_CLOSE
MASTER2PUPPET_JOINT_FN = lambda x: PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN(MASTER_GRIPPER_JOINT_NORMALIZE_FN(x))

MASTER_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE)
PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE)

MASTER_POS2JOINT = lambda x: MASTER_GRIPPER_POSITION_NORMALIZE_FN(x) * (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE) + MASTER_GRIPPER_JOINT_CLOSE
MASTER_JOINT2POS = lambda x: MASTER_GRIPPER_POSITION_UNNORMALIZE_FN((x - MASTER_GRIPPER_JOINT_CLOSE) / (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE))
PUPPET_POS2JOINT = lambda x: PUPPET_GRIPPER_POSITION_NORMALIZE_FN(x) * (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE) + PUPPET_GRIPPER_JOINT_CLOSE
PUPPET_JOINT2POS = lambda x: PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN((x - PUPPET_GRIPPER_JOINT_CLOSE) / (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE))

MASTER_GRIPPER_JOINT_MID = (MASTER_GRIPPER_JOINT_OPEN + MASTER_GRIPPER_JOINT_CLOSE)/2
