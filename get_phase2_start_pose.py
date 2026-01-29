
import numpy as np
from piper_ee_sim_env import make_ee_sim_env
from scripted_policy import VariableCoopPolicy
from piper_constants import PUPPET_GRIPPER_POSITION_NORMALIZE_FN
import IPython

def main():
    # Phase 2 start time (from process_coop_data.py default)
    SPLIT_STEP = 260 
    
    # Create Env (Phase 1 config)
    env = make_ee_sim_env('sim_variable_coop')
    ts = env.reset()
    
    policy = VariableCoopPolicy(inject_noise=False)
    
    print(f"Running VariableCoopPolicy for {SPLIT_STEP} steps to reach Phase 2 start...")
    
    for t in range(SPLIT_STEP):
        action = policy(ts)
        ts = env.step(action)
        
    print(f"\nreached t={SPLIT_STEP}")
    
    # Get qpos
    # Note: sim_variable_coop env doesn't return raw qpos in observation['qpos'] directly like sim_env?
    # piper_ee_sim_env wraps piper_sim_env.
    # Let's check accessing physics directly.
    physics = env.physics
    qpos_raw = physics.data.qpos.copy()
    
    # Extract Robot qpos (first 14 dims match action space structure? No.
    # piper_sim_env: qpos [0:6] Left Arm, [6] Left Grip, [8:14] Right Arm, [14] Right Grip.
    # Total robot qpos in physics: 16 (including dummy grippers) or 18?
    # piper_sim_env.get_qpos returns 14 dims.
    
    from piper_sim_env import BimanualPiperTask
    qpos_14 = BimanualPiperTask.get_qpos(physics)
    
    print("\n--- Handover Pose (t=280) ---")
    print("np.array(" + np.array2string(qpos_14, separator=', ', precision=4, suppress_small=True) + ")")
    print("-----------------------------\n")

    # Also print gripper states specifically
    print(f"Left Gripper: {qpos_14[6]}")
    print(f"Right Gripper: {qpos_14[13]}")

if __name__ == '__main__':
    main()
