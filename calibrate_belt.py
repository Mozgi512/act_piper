
import numpy as np
import time
from piper_ee_sim_env import make_ee_sim_env
from piper_constants import BELT_MOVE_SPEED, DT
from piper_sim_env import MANYCUBES_COLORS
import scripted_policy # To access CALIBRATION_OFFSET if needed, or just hardcode checking

def main():
    print("=== Belt Calibration Start ===")
    
    # Inject colors to trigger correct spawn logic (0.00, -0.15...)
    MANYCUBES_COLORS[0] = ['r'] * 10
    
    env = make_ee_sim_env('sim_many_cubes')
    ts = env.reset()
    
    # Cube 0 is at the beginning of env_state (first 7 elements)
    # env_state structure: 10 cubes * 7 dims
    
    # Initial Position
    cube0_state = ts.observation['env_state'][0:7]
    start_x = cube0_state[0]
    print(f"Start Step 0: Cube 0 X = {start_x:.6f}")
    
    # Constants
    belt_speed_const = BELT_MOVE_SPEED # 0.03
    calib_offset = getattr(scripted_policy, 'CALIBRATION_OFFSET', 0.0)
    print(f"Config: BELT_MOVE_SPEED={belt_speed_const}, DT={DT}, CALIBRATION_OFFSET={calib_offset}")
    
    # Simple Action (Stationary)
    # action size for ee env: 
    # left_arm (7), left_grip (1), right_arm (7), right_grip (1) -> 16
    # But wait, make_ee_sim_env wraps BimanualPiperEETask.
    # before_step takes action of size 16.
    
    # Lets just hold current pose
    mocap_l = ts.observation['mocap_pose_left']
    mocap_r = ts.observation['mocap_pose_right']
    # Gripper 1 (Open)
    action = np.concatenate([mocap_l, [1.0], mocap_r, [1.0]])
    
    history_x = []
    
    steps = 50
    for i in range(1, steps + 1):
        ts = env.step(action)
        cube_x = ts.observation['env_state'][0] # Cube 0 X
        history_x.append(cube_x)
        
        # Prediction check
        # Formula: Target = Start + Speed * Time + Offset
        # Time = i * DT
        # Speed: +BELT_MOVE_SPEED (as currently configured)
        
        pred_x_pos = start_x + belt_speed_const * (i * DT) + calib_offset
        pred_x_neg = start_x - belt_speed_const * (i * DT) + calib_offset
        
        diff_pos = cube_x - pred_x_pos
        diff_neg = cube_x - pred_x_neg
        
        print(f"Step {i}: Actual={cube_x:.6f} | Pred(+X)={pred_x_pos:.6f} (Diff={diff_pos:.6f}) | Pred(-X)={pred_x_neg:.6f} (Diff={diff_neg:.6f})")

    # Summary
    end_x = history_x[-1]
    total_disp = end_x - start_x
    duration = steps * DT # 1.0 sec
    measured_speed = total_disp / duration
    
    print("\n=== Results ===")
    print(f"Total Displacement (1.0s): {total_disp:.6f} m")
    print(f"Measured Speed: {measured_speed:.6f} m/s")
    print(f"Configured Speed: {belt_speed_const} m/s")
    
    if abs(measured_speed - belt_speed_const) < 0.001:
        print(">> Speed Matches Config (+X direction)")
    elif abs(measured_speed - (-belt_speed_const)) < 0.001:
        print(">> Speed Matches Config (-X direction)")
    else:
        print(">> Speed Mismatch! Recommendation: Update BELT_MOVE_SPEED")

if __name__ == "__main__":
    main()
