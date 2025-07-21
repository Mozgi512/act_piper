import mujoco
import mujoco.viewer
import numpy as np
import time

# Helper: Check if arm reached target
def is_at_target(current_pos, target_pos, tol=1e-2):
    """Checks if the Euclidean distance between two points is within a tolerance."""
    return np.linalg.norm(current_pos - target_pos) < tol

# Helper: Reset box velocities
def zero_box_velocity(data, model):
    """Sets the velocity of the 'box' body to zero."""
    box_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "box")
    if box_body_id != -1:
        # The free joint of the box has 6 degrees of freedom (3 trans, 3 rot)
        box_qvel_adr = model.jnt_dofadr[model.body_jntadr[box_body_id]]
        data.qvel[box_qvel_adr:box_qvel_adr+6] = 0

# --- Main simulation ---

# Load the MuJoCo model
try:
    model = mujoco.MjModel.from_xml_path("bimanual_piper_ee_transfer_cube.xml")
except FileNotFoundError:
    print("エラー: 'bimanual_piper_ee_transfer_cube.xml' が見つかりません。")
    exit()

data = mujoco.MjData(model)

# --- Setup ---

# Get mocap and hand body indices
mocap_left_id = 0
mocap_right_id = 1
left_hand_id = model.body('l_weldpoint').id
right_hand_id = model.body('r_weldpoint').id

# Define gripper control values and set initial state
GRIPPER_OPEN_CTRL = 0.04
GRIPPER_CLOSE_CTRL = 0.021
data.ctrl[0] = GRIPPER_OPEN_CTRL
data.ctrl[1] = GRIPPER_OPEN_CTRL

# Define key positions and heights
orig_box_pos = np.array([0.2, 0, 0.225])
new_box_pos = np.array([0, 0, 0.225])
GRIP_Z_HEIGHT = 0.085
CLEARANCE_Z_HEIGHT = GRIP_Z_HEIGHT + 0.05

# --- Main Loop ---

with mujoco.viewer.launch_passive(model, data) as viewer:
    print("シミュレーション開始。")

    # Store initial mocap positions for resetting arms
    mocap_right_init = np.copy(data.mocap_pos[mocap_right_id])
    mocap_left_init = np.copy(data.mocap_pos[mocap_left_id])
    
    # Define specific arm targets based on key positions
    left_arm_above_orig_box = np.array([orig_box_pos[0], orig_box_pos[1] + 0.05, CLEARANCE_Z_HEIGHT])
    left_arm_at_orig_box = np.array([orig_box_pos[0], orig_box_pos[1] + 0.05, GRIP_Z_HEIGHT])
    left_arm_above_new_box = np.array([new_box_pos[0], new_box_pos[1] + 0.05, CLEARANCE_Z_HEIGHT])
    left_arm_at_new_box = np.array([new_box_pos[0], new_box_pos[1] + 0.05, GRIP_Z_HEIGHT])

    right_arm_above_new_box = np.array([new_box_pos[0], new_box_pos[1] - 0.05, CLEARANCE_Z_HEIGHT])
    right_arm_at_new_box = np.array([new_box_pos[0], new_box_pos[1] - 0.05, GRIP_Z_HEIGHT])
    right_arm_above_orig_box = np.array([orig_box_pos[0], orig_box_pos[1] - 0.05, CLEARANCE_Z_HEIGHT])
    right_arm_at_orig_box = np.array([orig_box_pos[0], orig_box_pos[1] - 0.05, GRIP_Z_HEIGHT])

    phase = 1

    while viewer.is_running():

        # === Get current hand positions ===
        left_hand_pos = data.body(left_hand_id).xpos
        right_hand_pos = data.body(right_hand_id).xpos

        # Phase 1: Idle state, arms at home, grippers open
        if phase == 1:
            data.mocap_pos[mocap_right_id] = mocap_right_init
            data.mocap_pos[mocap_left_id] = mocap_left_init
            data.ctrl[0] = GRIPPER_OPEN_CTRL
            data.ctrl[1] = GRIPPER_OPEN_CTRL

            # Wait for BOTH arms to settle at their home positions
            if is_at_target(left_hand_pos, mocap_left_init) and \
               is_at_target(right_hand_pos, mocap_right_init):
                print("[INFO] Phase 1 -> 2")
                phase = 2
                time.sleep(1)

        # Phase 2: Left arm moves above the box
        elif phase == 2:
            data.mocap_pos[mocap_left_id] = left_arm_above_orig_box
            if is_at_target(left_hand_pos, left_arm_above_orig_box):
                print("[INFO] Phase 2 -> 3")
                phase = 3
                time.sleep(1)

        # Phase 3: Left arm lowers to the box
        elif phase == 3:
            data.mocap_pos[mocap_left_id] = left_arm_at_orig_box
            if is_at_target(left_hand_pos, left_arm_at_orig_box, tol=5e-3):
                print("[INFO] Phase 3 -> 4")
                phase = 4
                time.sleep(1)

        # Phase 4: Close left gripper to grab the cube
        elif phase == 4:
            data.ctrl[0] = GRIPPER_CLOSE_CTRL
            # Assuming gripper closing takes some time
            if data.time > 1: # Simple time-based wait for gripper
                zero_box_velocity(data, model)
                print("[INFO] Phase 4 -> 5")
                phase = 5
                time.sleep(1)

        # Phase 5: Lift and move left arm to the new position
        elif phase == 5:
            data.mocap_pos[mocap_left_id] = left_arm_above_new_box
            if is_at_target(left_hand_pos, left_arm_above_new_box):
                print("[INFO] Phase 5 -> 6")
                phase = 6
                time.sleep(1)

        # Phase 6: Lower left arm at the new position
        elif phase == 6:
            data.mocap_pos[mocap_left_id] = left_arm_at_new_box
            if is_at_target(left_hand_pos, left_arm_at_new_box):
                print("[INFO] Phase 6 -> 7")
                phase = 7
                time.sleep(1)

        # Phase 7: Right arm moves above the new box position
        elif phase == 7:
            data.mocap_pos[mocap_right_id] = right_arm_above_new_box
            if is_at_target(right_hand_pos, right_arm_above_new_box):
                print("[INFO] Phase 7 -> 8")
                phase = 8
                time.sleep(1)

        # Phase 8: Right arm lowers to the box
        elif phase == 8:
            data.mocap_pos[mocap_right_id] = right_arm_at_new_box
            if is_at_target(right_hand_pos, right_arm_at_new_box):
                print("[INFO] Phase 8 -> 9")
                phase = 9
                time.sleep(1)

        # Phase 9: Close right gripper and detach left gripper
        elif phase == 9:
            data.ctrl[1] = GRIPPER_CLOSE_CTRL
            data.ctrl[0] = GRIPPER_OPEN_CTRL # Open left gripper
            if data.time > 1:
                zero_box_velocity(data, model)
                print("[INFO] Phase 9 -> 10")
                phase = 10
                time.sleep(1)

        # Phase 10: Lift and move right arm back to the original spot
        elif phase == 10:
            data.mocap_pos[mocap_right_id] = right_arm_above_orig_box
            if is_at_target(right_hand_pos, right_arm_above_orig_box):
                print("[INFO] Phase 10 -> 11")
                phase = 11
                time.sleep(1)
        
        # Phase 11: Lower right arm to the drop spot
        elif phase == 11:
            data.mocap_pos[mocap_right_id] = right_arm_at_orig_box
            if is_at_target(right_hand_pos, right_arm_at_orig_box):
                print("[INFO] Phase 11 -> 12")
                phase = 12
                time.sleep(1)

        # Phase 12: Open right gripper to release the box and reset
        elif phase == 12:
            data.ctrl[1] = GRIPPER_OPEN_CTRL
            if data.time > 1:
                zero_box_velocity(data, model)
                print("[INFO] Phase 12 -> 1. Resetting cycle.")
                phase = 1
                time.sleep(2)

        mujoco.mj_step(model, data)
        viewer.sync()