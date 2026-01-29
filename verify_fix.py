import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
from interactive_policy import InteractivePolicy
from piper_ee_sim_env import make_ee_sim_env

def test_immediate_return():
    print("Initialize...")
    env = make_ee_sim_env('sim_many_cubes')
    ts = env.reset()
    policy = InteractivePolicy(env)
    policy.init_pose(ts)
    
    # Wait for env to settle
    for _ in range(50):
        ts = env.step(policy(ts))
        
    print("Scheduling Cooperative Task...")
    policy.schedule_command('C', ts)
    
    if not policy.left_trajectory or not policy.right_trajectory:
        print("Error: No trajectory")
        sys.exit(1)
        
    print("Trajectory Lengths: L={}, R={}".format(len(policy.left_trajectory), len(policy.right_trajectory)))
    # Previous (with clearance): L=17, R=12 (depends on which arm places?)
    # Expect -1 waypoint each (Clearance step removed)
        
    print("Running simulation...")
    # Run through the pick sequence (approx 300 steps)
    for i in range(300):
        action = policy(ts)
        ts = env.step(action)
        
    print("Test Passed (Code Runs).")

if __name__ == "__main__":
    test_immediate_return()
