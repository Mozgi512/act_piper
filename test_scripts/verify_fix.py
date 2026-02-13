
import numpy as np
import collections
from interactive_policy import InteractivePolicy

# Mock Observation
def get_mock_ts():
    obs = collections.OrderedDict()
    obs['qpos'] = np.zeros(14)
    obs['mocap_pose_left'] = np.array([0, 0.3, 0.2, 0, 0, 0, 1])
    obs['mocap_pose_right'] = np.array([0, -0.3, 0.2, 0, 0, 0, 1])
    obs['env_state'] = np.zeros(70) # 10 cubes
    # Place Red (0) at x=0.4 (Right side)
    obs['env_state'][0:3] = [0.4, 0.3, 0.02] 
    
    # Mock TS object
    TimeStep = collections.namedtuple("TimeStep", ["observation"])
    return TimeStep(observation=obs)

def verify():
    print("Testing InteractivePolicy Trajectory Smoothness...")
    policy = InteractivePolicy(inject_noise=False)
    ts = get_mock_ts()
    
    # Init
    policy.init_pose(ts)
    
    # Simulate 500 steps of idle
    print("Stepping 500 steps (Idle)...")
    for _ in range(500):
        policy(ts)
        # policy.step_count += 1 # Removed double increment
        
    print(f"Current Step: {policy.step_count}")
    
    # Inject Command 'R'
    print("Injecting Command 'R'...")
    policy.schedule_command('R')
    
    # Move object to reachable zone for Right arm (> -0.05)
    # env_state index 0 (Red X) -> Change to -0.01
    ts.observation['env_state'][0] = -0.01 
    
    # Process buffer
    policy.process_command_buffer(ts)
    
    # Check Right Trajectory
    traj = policy.right_trajectory
    print(f"Right Trajectory has {len(traj)} waypoints:")
    for i, wp in enumerate(traj):
        if i > 5: break # Show first few
        print(f"  WP {i}: t={wp['t']}, xyz={wp['xyz']}")
        
    # Verify Gap
    # We expect:
    # WP 0: t=0 (Init)
    # WP 1: t=Start (Bridged) -> Matches Init
    # WP 2: t=Intercept (Target)
    
    if len(traj) < 3:
        print("FAILURE: Trajectory too short. Expected bridge waypoint.")
        return

    wp_prev = traj[-4] if len(traj) > 3 else traj[0] # The one before the new action
    # Actually, let's look at the sequence
    # 0: t=0
    # 1: t=520 (Bridge) <-- THIS IS WHAT WE WANT
    # 2: t=610 (Intercept)
    
    bridge_wp = None
    for wp in traj:
        if wp['t'] == 520:
            bridge_wp = wp
            break
            
    if bridge_wp:
        print("\nSUCCESS: Found bridge waypoint at t=520.")
        print(f"  Bridge XYZ: {bridge_wp['xyz']}")
        print(f"  Init XYZ: {policy.init_right_pose['xyz']}")
        
        diff = np.linalg.norm(bridge_wp['xyz'] - policy.init_right_pose['xyz'])
        if diff < 1e-6:
            print("  Bridge waypoint matches previous pose (Smooth).")
        else:
            print(f"  FAILURE: Bridge waypoint pose mismatch! Diff: {diff}")
    else:
        print("\nFAILURE: No bridge waypoint found at t=520.")
        print("Waypoints found:")
        for wp in traj:
            print(f"  t={wp['t']}")

if __name__ == "__main__":
    verify()
