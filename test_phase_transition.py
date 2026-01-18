
import numpy as np
import os
import collections
from piper_sim_env import make_sim_env
from scripted_policy import IndependentPolicy, IndependentPhase2Policy

def get_cube_pos(physics, cube_idx):
    # Joint name: cube_{i}_joint
    # id = physics.model.name2id(...)
    # qpos_adr = ...
    # But qpos has 7 dims. X,Y,Z is first 3.
    joint_name = f"cube_{cube_idx}_joint"
    joint_id = physics.model.name2id(joint_name, 'joint')
    qpos_adr = physics.model.jnt_qposadr[joint_id]
    return physics.data.qpos[qpos_adr:qpos_adr+3].copy()

def test_transition():
    import matplotlib.pyplot as plt
    from piper_constants import DT

    onscreen_render = True

    # --- Phase 1 ---
    print("\n[Phase 1] Running sim_independent...")
    # IndependentPolicy requires EE obs (mocap_pose), so use make_ee_sim_env
    from piper_ee_sim_env import make_ee_sim_env
    env1 = make_ee_sim_env('sim_independent')
    ts1 = env1.reset()
    policy1 = IndependentPolicy(inject_noise=False)
    
    if onscreen_render:
        ax = plt.subplot()
        cam_image = env1.physics.render(height=360, width=640, camera_id="top")
        plt_img = ax.imshow(cam_image)
        plt.ion()

    # Run for 360 steps
    for t in range(360):
        action = policy1(ts1)
        ts1 = env1.step(action)
        if onscreen_render:
            cam_image = env1.physics.render(height=360, width=640, camera_id="top")
            plt_img.set_data(cam_image)
            plt.pause(DT)

    
    # Capture End State
    p1_red = get_cube_pos(env1.physics, 7)
    p1_green = get_cube_pos(env1.physics, 8)
    p1_blue = get_cube_pos(env1.physics, 9)
    print(f"Phase 1 End:")
    print(f"  Red   (7): {p1_red}")
    print(f"  Green (8): {p1_green}")
    print(f"  Blue  (9): {p1_blue}")
    
    env1.close()
    
    # --- Phase 2 ---
    print("\n[Phase 2] Initializing sim_independent_phase2_scripted...")
    # Note: Phase 2 init uses random sampling, so it might not match EXACTLY if randomness is involved.
    # But we want to see if the MEAN matches or if logic aligns.
    # Use EE env for policy compatibility
    env2 = make_ee_sim_env('sim_independent_phase2_scripted')
    ts2 = env2.reset()
    
    if onscreen_render:
        # Re-init plot for Phase 2? Or reuse?
        # Simpler to just re-init or continue. 
        # But ax belongs to plt figure. 
        # We can just update the image data with new env physics.
        pass

    # Capture Start State
    p2_red = get_cube_pos(env2.physics, 7)
    p2_green = get_cube_pos(env2.physics, 8)
    p2_blue = get_cube_pos(env2.physics, 9)
    print(f"Phase 2 Start:")
    print(f"  Red   (7): {p2_red}")
    print(f"  Green (8): {p2_green}")
    print(f"  Blue  (9): {p2_blue}")
    
    if onscreen_render:
        # Show Phase 2 start
        cam_image = env2.physics.render(height=360, width=640, camera_id="top")
        plt_img.set_data(cam_image)
        plt.pause(0.5) 
    
    policy2 = IndependentPhase2Policy(inject_noise=False)
    print("Running Phase 2...")
    for t in range(360):
        action = policy2(ts2)
        ts2 = env2.step(action)
        if onscreen_render:
            cam_image = env2.physics.render(height=360, width=640, camera_id="top")
            plt_img.set_data(cam_image)
            plt.pause(DT)

    env2.close()
    
    # --- Comparison ---
    print("\n[Comparison]")
    print(f"Red Gap:   {np.linalg.norm(p1_red - p2_red):.4f} (Vector: {p2_red - p1_red})")
    print(f"Green Gap: {np.linalg.norm(p1_green - p2_green):.4f} (Vector: {p2_green - p1_green})")
    print(f"Blue Gap:  {np.linalg.norm(p1_blue - p2_blue):.4f} (Vector: {p2_blue - p1_blue})")

if __name__ == "__main__":
    test_transition()
