
import numpy as np
from dm_control import mujoco
import matplotlib.pyplot as plt
from piper_ee_sim_env import make_ee_sim_env
from piper_sim_env import make_sim_env
from piper_constants import XML_DIR

def get_cube_pos(physics, i):
    name = f'cube_{i}'
    return physics.named.data.xpos[name].copy()

def main():
    seed = 42
    np.random.seed(seed)
    
    # EE Env
    print("Initializing EE Env...")
    np.random.seed(seed) # Reset seed before
    env_ee = make_ee_sim_env('sim_independent_phase2_scripted')
    ts_ee = env_ee.reset()
    ee_poses = [get_cube_pos(env_ee.physics, i) for i in range(10)]
    env_ee.close()
    
    # Joint Env
    print("Initializing Joint Env...")
    np.random.seed(seed) # Reset seed before
    env_joint = make_sim_env('sim_independent_phase2_scripted')
    ts_joint = env_joint.reset()
    joint_poses = [get_cube_pos(env_joint.physics, i) for i in range(10)]
    env_joint.close()
    
    print("\nComparison (Diff > 1e-4):")
    for i in range(10):
        diff = np.linalg.norm(ee_poses[i] - joint_poses[i])
        print(f"Cube {i}: EE={ee_poses[i]} Joint={joint_poses[i]} Diff={diff:.6f}")

if __name__ == "__main__":
    main()
