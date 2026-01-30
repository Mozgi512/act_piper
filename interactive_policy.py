import time
import sys
import select
import os
os.environ['MUJOCO_GL'] = 'egl'

import numpy as np
from pyquaternion import Quaternion

from piper_ee_sim_env import make_ee_sim_env
from piper_sim_env import MANYCUBES_COLORS
from piper_constants import BELT_MOVE_SPEED
from scripted_policy import VariableCoopPolicy

import IPython
e = IPython.embed

# Predefined Sequence (R:G:B = ~2:1:1) -> 5R, 3G, 2B
COLOR_SEQUENCE = ['r', 'r', 'g', 'b', 'r', 'r', 'g', 'b', 'r', 'g']

class InteractivePolicy(VariableCoopPolicy):
    def __init__(self, inject_noise=False):
        super().__init__(inject_noise)
        self.left_trajectory = [
            {"t": 0, "xyz": [0,0,0], "quat": [1,0,0,0], "gripper": 1},
            {"t": 100000, "xyz": [0,0,0], "quat": [1,0,0,0], "gripper": 1} 
        ]
        self.right_trajectory = [
            {"t": 0, "xyz": [0,0,0], "quat": [1,0,0,0], "gripper": 1},
            {"t": 100000, "xyz": [0,0,0], "quat": [1,0,0,0], "gripper": 1} 
         ]
        self.step_count = 0
        self.initialized = False
        
        self.picked_objects = set()
        self.sequence_idx = 0 # Points to the *next* object in COLOR_SEQUENCE to process
        self.task_count = 0  # Counter for Y-offset calculation (5cm per task) - DEPRECATED, use specific counters
        self.last_action_end_t = -1

    def generate_trajectory(self, ts_first):
        self.init_pose(ts_first)

    def init_pose(self, ts):
        if self.initialized: return
        
        init_mocap_pose_right = ts.observation['mocap_pose_right']
        init_mocap_pose_left = ts.observation['mocap_pose_left']
        
        # Store initial poses for returning home after tasks
        self.init_left_pose = {"xyz": init_mocap_pose_left[:3].copy(), "quat": init_mocap_pose_left[3:].copy()}
        self.init_right_pose = {"xyz": init_mocap_pose_right[:3].copy(), "quat": init_mocap_pose_right[3:].copy()}
        
        self.left_trajectory[0] = {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 1}
        self.left_trajectory[1] = {"t": 100000, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 1}
        
        self.right_trajectory[0] = {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 1}
        self.right_trajectory[1] = {"t": 100000, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 1}
        
        self.initialized = True
        print("Policy Initialized. Sequence:", COLOR_SEQUENCE)

    def scan_conveyor(self, ts, color_filter=None):
        """Finds objects on the conveyor (X < 0.35) matching color_filter."""
        work_info = np.array(ts.observation['env_state'])
        # In sim_four_objects (ManyCubes), env_state is just the cubes (70 floats)
        cubes_state = work_info 
        
        available = [] # (index, x_pos)
        
        for i in range(10): # 10 cubes
            if i in self.picked_objects: continue
            
            # Determine color from static sequence
            if i >= len(COLOR_SEQUENCE): break
            color = COLOR_SEQUENCE[i]
            
            if color_filter and color != color_filter: continue
            
            offset = i * 7
            xyz = cubes_state[offset : offset+3]
            
            # Check if on conveyor (rough bounds)
            # Spawn is around 0.2 to 0.4. Belt moves to -X.
            # "On conveyor" implies reachable and moving.
            # Restrict X range to avoid over-reaching (causing singularities)
            if xyz[0] < 0.4 and xyz[0] > -0.45:
                available.append((i, xyz[0], color))
                
        # Sort by X descending (Rightmost first -> coming effectively?)
        # Actually items flow from right (+X) to left (-X).
        # We want to pick items that are reachable.
        # Sort by X descending (Rightmost first)
        available.sort(key=lambda x: x[1], reverse=True) 
        return available

    def get_last_waypoint(self, is_left):
        traj = self.left_trajectory if is_left else self.right_trajectory
        if len(traj) > 0:
            return traj[-1]
        
        # Fallback to current waypoint (maintained by BasePolicy)
        if is_left:
            if hasattr(self, 'curr_left_waypoint'): return self.curr_left_waypoint
        else:
            if hasattr(self, 'curr_right_waypoint'): return self.curr_right_waypoint
            
        # Fallback to init pose reference if even current is missing (unlikely)
        return {"t": 0, "xyz": [0,0,0], "quat": [1,0,0,0], "gripper": 1}

    def schedule_command(self, cmd, ts):
        if not self.initialized: self.init_pose(ts)
        
        work_info = np.array(ts.observation['env_state'])
        cubes_state = work_info
        
        if cmd == 'I': # Independent: Pick next 2 Reds
            # Look for next 2 Reds in sequence starting from current check
            # BUT user said "Independent left and right pick NEXT TWO RED OBJECTS"
            # Does this mean from global sequence?
            # "现時点コンベア上に乗っている物体だけを次に掴む目標物体とする"
            # Target only objects currently on conveyor.
            
            # Find Reds on conveyor
            reds = self.scan_conveyor(ts, 'r')
            
            if len(reds) < 2:
                print(f"Not enough Reds on conveyor! Found {len(reds)}")
                return
            
            # Pick first 2 available Reds
            r1 = reds[0] # Right-most
            r2 = reds[1] # Next Right-most
            
            print(f"Independent: Scheduling Red {r1[0]} (R-Arm) and Red {r2[0]} (L-Arm)")
            # Note: Heuristic logic. Right-most (Highest X) to Right Arm?
            # Let's assign based on X position like previously.
            
            targets = [r1, r2]
            # Remove from tracking
            self.picked_objects.add(r1[0])
            self.picked_objects.add(r2[0])
            
            self.plan_independent_pick(targets, cubes_state)

        elif cmd == 'C': # Cooperative: Assemble G and B
            greens = self.scan_conveyor(ts, 'g')
            blues = self.scan_conveyor(ts, 'b')
            
            if not greens or not blues:
                print(f"Missing parts for Assembly! G:{len(greens)}, B:{len(blues)}")
                return
                
            g = greens[0]
            b = blues[0]
            
            print(f"Cooperative: Scheduling Assembly G:{g[0]} + B:{b[0]}")
            self.picked_objects.add(g[0])
            self.picked_objects.add(b[0])
            
            self.plan_cooperative_assembly(g, b, cubes_state)
            
    def plan_independent_pick(self, targets, cubes_state):
        # targets: list of (idx, x, color)
        # Assign to arms based on X split
        # Simply: Right-most -> Right Arm, Next -> Left Arm?
        # Or split at center 0.0?
        
        # Remove tails
        if self.left_trajectory: self.left_trajectory.pop()
        if self.right_trajectory: self.right_trajectory.pop()
        
        last_l = self.get_last_waypoint(is_left=True)
        last_r = self.get_last_waypoint(is_left=False)
        
        t_left_end = last_l['t']
        t_right_end = last_r['t']
        
        belt_speed = BELT_MOVE_SPEED
        # Calculate Y-offset: 5cm per Independent task to prevent collisions
        # Base position is 0.05m, so 1st task: 0.05m, 2nd: 0.10m, 3rd: 0.15m...
        goal_xyz = np.array([0, 0.10, 0.025])
        
        # Sort targets by X descending
        targets.sort(key=lambda x: x[1], reverse=True)
        
        # Assign Strategy:
        # If both X > 0: Right picks 1st, then Left picks 2nd (Cross?) or Right picks both?
        # "Left and Right each pick" implies split.
        # Let's force split: Left Arm takes one, Right Arm takes one.
        # Which one?
        # Right Arm is at +Y/+X side. Left Arm is at +Y/-X side.
        # Right Arm should take the one with larger X (Right-most).
        # Left Arm should take the one with smaller X.
        
        t_right_target = targets[0] # Max X
        t_left_target = targets[1]  # Min X
        
        # Right Arm Plan
        offset = t_right_target[0] * 7
        obj_xyz = cubes_state[offset : offset+3] # Still needed? No, logic moved to helper.
        # But we pass obj_idx = t_right_target[0]
        # Sync Start: Ensure we start from current time or last connection
        # Add 20 steps buffer to allow smooth transition from Loading/Holding to Reach
        start_t = max(t_right_end, self.step_count) + 20
        # If gap exists, fill it with hold (handled by get_last_waypoint/interpolate fallback?)
        # Better: Explicitly hold until start_t? 
        # Actually base policy interpolates. If we define start_t > last_t, it interpolates.
        if start_t > t_right_end:
             # Ensure last waypoint is preserved until start_t for smooth takeoff? 
             # No, simple interpolation from last_waypoint to first pick_waypoint is sufficient 
             # IF duration is long enough. 20 steps (0.4s) is good.
             pass

        self.add_pick_place(self.right_trajectory, start_t, t_right_target[0], goal_xyz, belt_speed, is_left=False, offset_x=0.08)
        
        # Left Arm Plan
        offset = t_left_target[0] * 7
        obj_xyz = cubes_state[offset : offset+3]
        start_t = max(t_left_end, self.step_count) + 20
        self.add_pick_place(self.left_trajectory, start_t, t_left_target[0], goal_xyz, belt_speed, is_left=True, offset_x=-0.08)

        # Return to home position
        return_t = max(self.left_trajectory[-1]['t'], self.right_trajectory[-1]['t']) + 60
        self.last_action_end_t = return_t
        self.left_trajectory.append({"t": return_t, "xyz": self.init_left_pose["xyz"], "quat": self.init_left_pose["quat"], "gripper": 1})
        self.right_trajectory.append({"t": return_t, "xyz": self.init_right_pose["xyz"], "quat": self.init_right_pose["quat"], "gripper": 1})
        
        # Restore Tails
        tail_t = max(self.left_trajectory[-1]['t'], self.right_trajectory[-1]['t']) + 10000
        self.left_trajectory.append({"t": tail_t, "xyz": self.left_trajectory[-1]['xyz'], "quat": self.left_trajectory[-1]['quat'], "gripper": self.left_trajectory[-1]['gripper']})
        self.right_trajectory.append({"t": tail_t, "xyz": self.right_trajectory[-1]['xyz'], "quat": self.right_trajectory[-1]['quat'], "gripper": self.right_trajectory[-1]['gripper']})

    def plan_cooperative_assembly(self, g_target, b_target, cubes_state):
        # G (Top) -> Green
        # B (Base) -> Blue
        
        if self.left_trajectory: self.left_trajectory.pop()
        if self.right_trajectory: self.right_trajectory.pop()
        
        last_l = self.get_last_waypoint(is_left=True)
        last_r = self.get_last_waypoint(is_left=False)
        
        t_left = last_l['t']
        t_right = last_r['t']
        
        # We need XYZ for comparison logic ONLY
        offset_g = g_target[0] * 7
        g_xyz = cubes_state[offset_g : offset_g+3]
        
        offset_b = b_target[0] * 7
        b_xyz = cubes_state[offset_b : offset_b+3]
        
        belt_speed = BELT_MOVE_SPEED
        meet_xyz = np.array([0, 0.3, 0.15]) 
        
        # Sync Start
        t_start = max(t_left, t_right, self.step_count)
        if t_left < t_start: 
             self.left_trajectory.append({"t": t_start, "xyz": last_l['xyz'], "quat": last_l['quat'], "gripper": last_l['gripper']})
        if t_right < t_start: 
             self.right_trajectory.append({"t": t_start, "xyz": last_r['xyz'], "quat": last_r['quat'], "gripper": last_r['gripper']})
        
        # Dynamic Assignment (Prevent Crossing)
        # Left Arm -> Left (Min X), Right Arm -> Right (Max X)
        # Role assignment: Green holder = Top, Blue holder = Base
        if g_xyz[0] < b_xyz[0]:
            # Left gets Green (Top), Right gets Blue (Base)
            self.add_pick_top(self.left_trajectory, t_start, g_target[0], meet_xyz, belt_speed, is_left=True)
            self.add_pick_base(self.right_trajectory, t_start, b_target[0], meet_xyz, None, belt_speed, is_left=False)
            left_holds_green = True  # Left has green = Left is Top
        else:
            # Left gets Blue (Base), Right gets Green (Top)
            self.add_pick_base(self.left_trajectory, t_start, b_target[0], meet_xyz, None, belt_speed, is_left=True)
            self.add_pick_top(self.right_trajectory, t_start, g_target[0], meet_xyz, belt_speed, is_left=False)
            left_holds_green = False  # Right has green = Right is Top

        # Sync at Meet
        t_l_mid = self.left_trajectory[-1]['t']
        t_r_mid = self.right_trajectory[-1]['t']
        t_assemble_start = max(t_l_mid, t_r_mid)
        
        if t_l_mid < t_assemble_start: self.add_wait(self.left_trajectory, t_assemble_start)
        if t_r_mid < t_assemble_start: self.add_wait(self.right_trajectory, t_assemble_start)
        
        # Assemble
        duration_assemble = 60
        t_done = t_assemble_start + duration_assemble
        
        q_l = self.left_trajectory[-1]['quat']
        q_r = self.right_trajectory[-1]['quat']
        
        # Top Arm does Insertion (Down) -> Release -> Up
        # Base Arm Holds -> Wait -> Carry
        # Green holder is ALWAYS Top (inserting), Blue holder is ALWAYS Base (holding)
        
        if left_holds_green:
            # Left has Green = Left is Top (Insertion)
            self.left_trajectory.append({"t": t_done, "xyz": meet_xyz + [0,0,0.04], "quat": q_l, "gripper": 1}) # Insert
            self.right_trajectory.append({"t": t_done, "xyz": meet_xyz, "quat": q_r, "gripper": 0}) # Hold
            
            t_release = t_done + 20
            self.left_trajectory.append({"t": t_release, "xyz": meet_xyz + [0,0,0.04], "quat": q_l, "gripper": 1}) # Release
            self.right_trajectory.append({"t": t_release, "xyz": meet_xyz, "quat": q_r, "gripper": 0}) # Hold
            
            t_retreat = t_release + 30
            self.left_trajectory.append({"t": t_retreat, "xyz": meet_xyz + [0,0,0.1], "quat": q_l, "gripper": 1}) # Up
            self.right_trajectory.append({"t": t_retreat, "xyz": meet_xyz, "quat": q_r, "gripper": 0}) # Hold
            
            # Place (Right carries Blue base)
            t_place_start = t_retreat
            self.add_place(self.right_trajectory, t_place_start, np.array([0, 0.05, 0.025]), is_left=False)
        else:
            # Right has Green = Right is Top (Insertion)
            self.right_trajectory.append({"t": t_done, "xyz": meet_xyz + [0,0,0.04], "quat": q_r, "gripper": 1}) # Insert
            self.left_trajectory.append({"t": t_done, "xyz": meet_xyz, "quat": q_l, "gripper": 0}) # Hold
            
            t_release = t_done + 20
            self.right_trajectory.append({"t": t_release, "xyz": meet_xyz + [0,0,0.04], "quat": q_r, "gripper": 1}) # Release
            self.left_trajectory.append({"t": t_release, "xyz": meet_xyz, "quat": q_l, "gripper": 0}) # Hold
            
            t_retreat = t_release + 30
            self.right_trajectory.append({"t": t_retreat, "xyz": meet_xyz + [0,0,0.1], "quat": q_r, "gripper": 1}) # Up
            self.left_trajectory.append({"t": t_retreat, "xyz": meet_xyz, "quat": q_l, "gripper": 0}) # Hold
            
            # Place (Left carries Blue base)
            t_place_start = t_retreat
            self.add_place(self.left_trajectory, t_place_start, np.array([0, 0.05, 0.025]), is_left=True)
        
        # Increment Cooperative task counter for next placement        
        # Return to home position
        # Placing arm returns first, other arm returns later
        if left_holds_green:
            # Right arm placed the object, so it returns first
            right_return_t = self.right_trajectory[-1]['t'] + 40
            self.right_trajectory.append({"t": right_return_t, "xyz": self.init_right_pose["xyz"], "quat": self.init_right_pose["quat"], "gripper": 1})
            # Left arm returns later
            left_return_t = right_return_t -70
            self.left_trajectory.append({"t": left_return_t, "xyz": self.init_left_pose["xyz"], "quat": self.init_left_pose["quat"], "gripper": 1})
        else:
            # Left arm placed the object, so it returns first
            left_return_t = self.left_trajectory[-1]['t'] + 40
            self.left_trajectory.append({"t": left_return_t, "xyz": self.init_left_pose["xyz"], "quat": self.init_left_pose["quat"], "gripper": 1})
            # Right arm returns later
            right_return_t = left_return_t -70
            self.right_trajectory.append({"t": right_return_t, "xyz": self.init_right_pose["xyz"], "quat": self.init_right_pose["quat"], "gripper": 1})
        
        
        # Calculate completion time
        return_t = max(self.left_trajectory[-1]['t'], self.right_trajectory[-1]['t'])
        self.last_action_end_t = return_t
        
        # Restore Tails
        tail_t = max(self.left_trajectory[-1]['t'], self.right_trajectory[-1]['t']) + 10000
        self.left_trajectory.append({"t": tail_t, "xyz": self.left_trajectory[-1]['xyz'], "quat": self.left_trajectory[-1]['quat'], "gripper": self.left_trajectory[-1]['gripper']})
        self.right_trajectory.append({"t": tail_t, "xyz": self.right_trajectory[-1]['xyz'], "quat": self.right_trajectory[-1]['quat'], "gripper": self.right_trajectory[-1]['gripper']})


        
    def add_place(self, traj, start_t, goal_xyz, is_left):
        t = start_t + 60
        q = self.get_quat(is_left, 'pick')
        traj.append({"t": t, "xyz": goal_xyz + [0,0,0.05], "quat": q, "gripper": 0})
        traj.append({"t": t+30, "xyz": goal_xyz, "quat": q, "gripper": 0})
        traj.append({"t": t+50, "xyz": goal_xyz, "quat": q, "gripper": 1}) # Release
        traj.append({"t": t+70, "xyz": goal_xyz + [0,0,0.1], "quat": q, "gripper": 1}) # Up
        return t+70
        

    # Inject Sequence
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--commands', type=str, help='Sequence of commands (e.g. "ICI")', default="")
    args = parser.parse_args()
    
    command_queue = list(args.commands)
    # Next, trigger at step 20 for the first command
    next_trigger = 20 if command_queue else -1

    MANYCUBES_COLORS[0] = COLOR_SEQUENCE
    
    task_name = 'sim_many_cubes'
    env = make_ee_sim_env(task_name, camera_names=['top'])
    policy = InteractivePolicy(inject_noise=False)
    
    # Render setup
    import cv2
    window_name = "Interactive Policy"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    try:
        cv2.startWindowThread()
    except:
        pass

    # Init render
    ts = env.reset()
    
    # Show initial frame immediately
    img_rgb = ts.observation['images']['top']
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    cv2.imshow(window_name, img_bgr)
    cv2.waitKey(100) # Wait a bit longer for first frame
    
    policy.init_pose(ts)
    
    print("\n\n=== INTERACTIVE POLICY STARTED ===")
    print(f"Sequence: {COLOR_SEQUENCE}")
    print("Commands:")
    print("  'I': Independent Pick (Next 2 Reds)")
    print("  'C': Cooperative Assembly (Next Green & Blue)")
    print("  'q': Quit")
    
    step = 0
    try:
        while True:
            # Auto Execution Logic
            if command_queue and step == next_trigger:
                cmd = command_queue.pop(0).upper()
                print(f"Auto-executing command '{cmd}' at step {step}")
                policy.schedule_command(cmd, ts)
                # Next trigger will be set upon completion
                next_trigger = -1

            # Non-blocking stdin read
            if select.select([sys.stdin], [], [], 0.0)[0]:
                line = sys.stdin.readline().strip().upper()
                if line == 'Q':
                    break
                elif line in ['I', 'C']:
                    policy.schedule_command(line, ts)
                else:
                    print("Unknown command. Use I, C or Q.")
            
            action = policy(ts)
            ts = env.step(action)
            
            # Render every step or every N steps
            if step % 2 == 0: 
                # Get RGB image
                img_rgb = ts.observation['images']['top']
                # Convert to BGR for OpenCV
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                
                cv2.imshow(window_name, img_bgr)
                # Wait 1ms to process events
                key = cv2.waitKey(10) & 0xFF
                if key == ord('q'):
                    break
             
            if step == policy.last_action_end_t:
                print(f"Subtask completed at timestep {step}")
                if command_queue:
                    next_trigger = step + 20
                    print(f"Next task scheduled at step {next_trigger}")

            step += 1
            
    except KeyboardInterrupt:
        pass
    finally:
        # plt.close()
        cv2.destroyAllWindows()
        print("Interactive Policy Terminated")

if __name__ == '__main__':
    main()
