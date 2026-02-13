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
    def __init__(self, inject_noise=False, color_sequence=None):
        super().__init__(inject_noise)
        # Use provided sequence or default
        self.color_sequence = color_sequence if color_sequence is not None else COLOR_SEQUENCE
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
        
        # Asynchronous Scheduling State
        self.command_buffer = [] # Queue of commands ['L', 'R', 'C', ...]
        
        # Dynamic Role Assignment
        # Default: Left is Top (Green), Right is Base (Blue) until swapped
        self.top_arm = 'left' 
        self.base_arm = 'right'
        
        self.previous_buffer_state = []
        
        # Arm busy state tracking for completion logging
        self.left_was_busy = False
        self.right_was_busy = False
        
        # Task segment metadata for data processing
        self.left_segments = []  # List of {'start': step, 'end': step, 'type': 'independent'/'cooperative'}
        self.right_segments = []
        self.current_left_segment = None
        self.current_right_segment = None
        
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
        print("Policy Initialized. Sequence:", self.color_sequence)

    def scan_conveyor(self, ts, color_filter=None):
        """Finds objects on the conveyor (X < 0.35) matching color_filter."""
        work_info = np.array(ts.observation['env_state'])
        # In sim_four_objects (ManyCubes), env_state is just the cubes (70 floats)
        cubes_state = work_info 
        
        available = [] # (index, x_pos)
        
        for i in range(10): # 10 cubes
            if i in self.picked_objects: continue
            
            # Determine color from static sequence
            if i >= len(self.color_sequence): break
            color = self.color_sequence[i]
            
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

    def schedule_command(self, cmd, ts=None):
        """Adds a high-level command to the buffer."""
        # Normalize input
        cmd = cmd.upper()
        if cmd not in ['I', 'C', 'L', 'R', 'T', 'B']:
            print(f"Ignored unknown command: {cmd}")
            return

        print(f"Received Command: {cmd}")
        self.command_buffer.append(cmd)

    def is_arm_free(self, is_left, current_step):
        traj = self.left_trajectory if is_left else self.right_trajectory
        # Last waypoint time. If existing plan ends in future, arm is busy.
        # Note: We have a semantic 'tail' waypoint at traj[-1] at 100,000.
        # We must check traj[-2] for the actual end of the last action.
        if len(traj) > 1:
            end_t = traj[-2]['t']
            is_free = current_step >= end_t
            #if is_free and (self.left_was_busy if is_left else self.right_was_busy):
                # print(f"DEBUG: {'Left' if is_left else 'Right'} arm became free at step {current_step} (last_t={end_t})")
            return is_free
        return True # Only initial tail exists, arm is free

    def finalize(self, step_count):
        """Close any remaining open segments at the end of the episode."""
        if self.current_left_segment:
            print(f"[Step {step_count}] Finalizing Left Segment (Start={self.current_left_segment['start']})")
            self.current_left_segment['end'] = step_count
            self.left_segments.append(self.current_left_segment)
            self.current_left_segment = None
            
        if self.current_right_segment:
            print(f"[Step {step_count}] Finalizing Right Segment (Start={self.current_right_segment['start']})")
            self.current_right_segment['end'] = step_count
            self.right_segments.append(self.current_right_segment)
            self.current_right_segment = None

    def process_command_buffer(self, ts):
        if not self.command_buffer: 
            if self.previous_buffer_state:
                # print("DEBUG: Buffer Empty")
                self.previous_buffer_state = []
            return

        if self.command_buffer != self.previous_buffer_state:
            # print(f"DEBUG: Buffer Changed: {self.command_buffer}")
            # Use a copy to store state
            self.previous_buffer_state = list(self.command_buffer)
        
        # Check for task completion (arms transitioning from busy to free)
        left_free = self.is_arm_free(True, self.step_count)
        right_free = self.is_arm_free(False, self.step_count)
        
        if self.left_was_busy and left_free:
            print(f"[Step {self.step_count}] Left Arm Task Completed")
            self.left_was_busy = False
            # Close current segment
            if self.current_left_segment:
                self.current_left_segment['end'] = self.step_count
                self.left_segments.append(self.current_left_segment)
                self.current_left_segment = None
            
        if self.right_was_busy and right_free:
            print(f"[Step {self.step_count}] Right Arm Task Completed")
            self.right_was_busy = False
            # Close current segment
            if self.current_right_segment:
                self.current_right_segment['end'] = self.step_count
                self.right_segments.append(self.current_right_segment)
                self.current_right_segment = None
        
        # 1. Expand/Resolve phase (Head only? or as deep as possible?)
        # We need to resolve pending I/T/B to know which physical arm they use.
        # But role assignments change after C.
        # So we can only safe resolve up to the first C.
        
        # Let's just do a scan loop.
        i = 0
        while i < len(self.command_buffer):
            cmd = self.command_buffer[i]
            
            if cmd == 'I':
                # Expand I -> T, B at position i
                self.command_buffer.pop(i)
                self.command_buffer.insert(i, 'B')
                self.command_buffer.insert(i, 'T')
                continue # Re-process new i (T)
            
            if cmd == 'T':
                target = 'L' if self.top_arm == 'left' else 'R'
                self.command_buffer[i] = target
                continue # Re-process
                
            if cmd == 'B':
                target = 'L' if self.base_arm == 'left' else 'R'
                self.command_buffer[i] = target
                continue

            if cmd == 'C':
                # Barrier: Role assignments might change after C.
                # Stop resolving subsequent commands until C is executed/popped.
                break
            
            # If C, L, or R, move to next
            i += 1
            
        # 2. Execution Phase (Look-ahead)
        # We want to execute the first available valid command for each arm.
        # But we must respect order: If L1 is queued before L2, L2 cannot run before L1.
        # C blocks everything after it because it swaps roles? 
        # C needs both arms. So C blocks L and R after it.
        # L blocks L after it. R blocks R after it.
        
        blocked_l = False
        blocked_r = False
        
        indices_to_pop = []
        
        # Snapshot buffer state to avoid modification issues during iteration
        # We need to act on the live buffer though.
        # Let's iterate and collect actions.
        
        i = 0
        while i < len(self.command_buffer):
            cmd = self.command_buffer[i]
            
            if cmd == 'C':
                # C acts as a barrier.
                # If either arm is already busy/blocked by previous task in queue, C cannot start.
                if blocked_l or blocked_r:
                    # C is blocked. And C blocks everything after it.
                    break 
                
                # Check physical availability
                if self.is_arm_free(True, self.step_count) and self.is_arm_free(False, self.step_count):
                     # Check resources
                    greens = self.scan_conveyor(ts, 'g')
                    blues = self.scan_conveyor(ts, 'b')
                    if greens and blues:
                        print(f"[Step {self.step_count}] Scheduling Coop Assembly G:{greens[0][0]} + B:{blues[0][0]}")
                        self.picked_objects.add(greens[0][0])
                        self.picked_objects.add(blues[0][0])
                        
                        # Start cooperative segments for both arms BEFORE planning
                        # Will be updated with split point during plan_cooperative_assembly
                        self.current_left_segment = {'start': self.step_count, 'type': 'cooperative', 'top_arm': None, 'coop_split': None}
                        self.current_right_segment = {'start': self.step_count, 'type': 'cooperative', 'top_arm': None, 'coop_split': None}
                        
                        self.plan_cooperative_assembly(greens[0], blues[0], np.array(ts.observation['env_state']))
                        self.left_was_busy = True
                        self.right_was_busy = True
                        
                        # Remove C
                        self.command_buffer.pop(i)
                        # Don't increment i, next item shifts down
                        # C uses both arms, so we are done for this step
                        blocked_l = True
                        blocked_r = True
                        break
                    else:
                        # Resources missing. C waits. C blocks following.
                        blocked_l = True
                        blocked_r = True
                        break
                else:
                    # Physical arms busy.
                    blocked_l = True
                    blocked_r = True
                    break

            elif cmd == 'L':
                if blocked_l:
                    i += 1
                    continue
                
                # Check physical availability
                if self.is_arm_free(True, self.step_count):
                    reds = self.scan_conveyor(ts, 'r')
                    
                    # Strategy: Pick Right-most safe object to avoid skipping, BUT avoid Right Arm's target.
                    # Right Arm always takes reds[0] (global Right-most).
                    # Left Arm should take the Right-most candidate that is NOT reds[0].
            
                    # Left arm can reach slightly into the right side to avoid idle time
                    candidates = [r for r in reds if r[1] < -0.1]
            
                    if not candidates:
                         blocked_l = True
                         i += 1
                         continue
            
                    r_reserved_idx = None
                    if self.is_arm_free(False, self.step_count):
                        # Only reserve for Right arm if it's within its reach (X > -0.05)
                        if reds[0][1] > -0.05:
                            r_reserved_idx = reds[0][0]
            
                    target = None
                    for c in candidates:
                        if c[0] == r_reserved_idx:
                            continue
                        target = c
                        break # Take right-most non-reserved candidate
            
                    if target is None:
                        blocked_l = True 
                        i += 1
                        continue
                 
                    # Calculate actual last action time (not tail)
                    traj_l = self.left_trajectory
                    t_last_l = traj_l[-2]['t'] if len(traj_l) > 1 else 0
                    start_t = max(t_last_l, self.step_count) + 20
                    print(f"[Step {self.step_count}] Scheduling Left Pick (Red {target[0]}) -> Starts at Step {start_t}")
                    self.picked_objects.add(target[0])
                    self.plan_single_pick(target, is_left=True, cubes_state=np.array(ts.observation['env_state']))
                    self.left_was_busy = True
                    
                    # Start independent segment for left arm
                    self.current_left_segment = {'start': self.step_count, 'type': 'independent'}
                    self.command_buffer.pop(i)
                    # Do NOT increment i
                    blocked_l = True 
                    continue

                else:
                    blocked_l = True
                
                i += 1

            elif cmd == 'R':
                if blocked_r:
                    i += 1
                    continue
                
                if self.is_arm_free(False, self.step_count):
                    reds = self.scan_conveyor(ts, 'r')
                    # Right arm should not cross too far into left side
                    reds = [r for r in reds if r[1] > -0.05]
                    if reds:
                         target = reds[0] # Right-most
                         
                         # Calculate actual last action time (not tail)
                         traj_r = self.right_trajectory
                         t_last_r = traj_r[-2]['t'] if len(traj_r) > 1 else 0
                         start_t = max(t_last_r, self.step_count) + 20
                         print(f"[Step {self.step_count}] Scheduling Right Pick (Red {target[0]}) -> Starts at Step {start_t}")
                         self.picked_objects.add(target[0])
                         self.plan_single_pick(target, is_left=False, cubes_state=np.array(ts.observation['env_state']))
                         self.right_was_busy = True
                         
                         # Start independent segment for right arm
                         self.current_right_segment = {'start': self.step_count, 'type': 'independent'}
                         self.command_buffer.pop(i)
                         blocked_r = True
                         continue
                    else:
                         blocked_r = True
                else:
                    blocked_r = True
                    
                i += 1

            else:
                 # Should be resolved already
                 i += 1


    def plan_independent_pick(self, targets, cubes_state):
        # This method is deprecated by the new async scheduling.
        # The logic has been moved to plan_single_pick and process_command_buffer.
        pass
            
    def plan_single_pick(self, target, is_left, cubes_state):
        # target: (idx, x, color)
        
        # Remove tail
        traj = self.left_trajectory if is_left else self.right_trajectory
        if traj: traj.pop()
        
        last_wp = self.get_last_waypoint(is_left)
        t_last = last_wp['t']
        
        offset = target[0] * 7
        # target_xyz = cubes_state[offset : offset+3] 
        
        belt_speed = BELT_MOVE_SPEED
        # Fixed Goal for independent tasks? Or offset?
        # User output implies previous logic used [0, 0.10, 0.025]
        goal_xyz = np.array([0, 0.10, 0.025])
        
        # Start time: max(last_end, current) + buffer
        
        # FIX: Add bridge waypoint at current step to anchor trajectory
        # This prevents the interpolator from using a very old t_last to interpolate to start_t,
        # ensuring the robot holds its position until the new command starts.
        if self.step_count > t_last:
             traj.append({"t": self.step_count, "xyz": last_wp['xyz'], "quat": last_wp['quat'], "gripper": last_wp['gripper']})
             # Update t_last since we added a waypoint
             t_last = self.step_count

        start_t = max(t_last, self.step_count) + 20

        if t_last < start_t:
             traj.append({"t": start_t, "xyz": last_wp['xyz'], "quat": last_wp['quat'], "gripper": last_wp['gripper']})
        
        offset_x = -0.08 if is_left else 0.08
        
        t_end = self.add_pick_place(traj, start_t, target[0], goal_xyz, belt_speed, is_left, offset_x=offset_x)
        
        # Return Home
        return_t = t_end + 60
        init_pose = self.init_left_pose if is_left else self.init_right_pose
        traj.append({"t": return_t, "xyz": init_pose["xyz"], "quat": init_pose["quat"], "gripper": 1})
        
        # Restore Tail
        tail_t = return_t + 100000
        traj.append({"t": tail_t, "xyz": init_pose["xyz"], "quat": init_pose["quat"], "gripper": 1})

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
        
        # Role assignment: Green holder = Top
        # Logic: Assign Green to arm that is strictly CLOSER or based on X?
        # Original Logic: Left gets Green if G.x < B.x.
        # BUT we have `self.top_arm` state now!
        # Should we respect the state?
        # "Also, update who T and B point to after each Coop task."
        # The user implies dynamic role assignment based on the TASK itself, OR simply alternating?
        # "Initially check G/B positions to assign roles. After every C task, swap roles."
        # Dynamic Assignment (Prevent Crossing)
        # Left Arm -> Left (Min X), Right Arm -> Right (Max X)
        # Role assignment: Green holder = Top, Blue holder = Base
        
        # Use Geometry to decide assignment to prevent collision
        print(f"[Step {self.step_count}] Cooperative Assignment:")
        print(f"  Green X: {g_xyz[0]:.3f}, Blue X: {b_xyz[0]:.3f}")
        
        if g_xyz[0] < b_xyz[0]:
            # Green is to the Left of Blue
            print(f"  -> Green is Left. Assignment: Left=Top (Green), Right=Base (Blue)")
            # Left gets Green (Top), Right gets Blue (Base)
            self.add_pick_top(self.left_trajectory, t_start, g_target[0], meet_xyz, belt_speed, is_left=True)
            self.add_pick_base(self.right_trajectory, t_start, b_target[0], meet_xyz, None, belt_speed, is_left=False)
            left_holds_green = True  # Left has green = Left is Top
            
            # Update Role State
            self.top_arm = 'left'
            self.base_arm = 'right'
        else:
            # Blue is to the Left of Green
            print(f"  -> Blue is Left. Assignment: Left=Base (Blue), Right=Top (Green)")
            # Left gets Blue (Base), Right gets Green (Top)
            self.add_pick_base(self.left_trajectory, t_start, b_target[0], meet_xyz, None, belt_speed, is_left=True)
            self.add_pick_top(self.right_trajectory, t_start, g_target[0], meet_xyz, belt_speed, is_left=False)
            left_holds_green = False  # Right has green = Right is Top
            
            # Update Role State
            self.top_arm = 'right'
            self.base_arm = 'left'

        print(f"Cooperative Assignment: Top(G)={self.top_arm}, Base(B)={self.base_arm}")
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
            
            # --- New Logic: Both Return Home -> Wait -> Base Place ---
            t_home_start = t_retreat
            t_home_arrival = t_home_start + 60
            
            # Both return home (Base keeps holding)
            self.left_trajectory.append({"t": t_home_arrival, "xyz": self.init_left_pose["xyz"], "quat": self.init_left_pose["quat"], "gripper": 1})
            self.right_trajectory.append({"t": t_home_arrival, "xyz": self.init_right_pose["xyz"], "quat": self.init_right_pose["quat"], "gripper": 0})

            # Wait 20 steps
            t_wait_end = t_home_arrival + 20
            # self.add_wait(self.left_trajectory, t_wait_end) # Top Arm should NOT wait
            self.add_wait(self.right_trajectory, t_wait_end)

            # Place (Right carries Blue base)
            t_place_start = t_wait_end
            t_place_end = self.add_place(self.right_trajectory, t_place_start, np.array([0, 0.05, 0.025]), is_left=False)
            
            # Right returns home after placing
            right_return_t = t_place_end + 40
            self.right_trajectory.append({"t": right_return_t, "xyz": self.init_right_pose["xyz"], "quat": self.init_right_pose["quat"], "gripper": 1})
            
            # Record cooperative split point (middle of Top arm's 20-step wait at home)
            # Store in current segments which will be closed when task completes
            coop_split_step = t_home_arrival + 10
            if self.current_left_segment:
                self.current_left_segment['top_arm'] = 'left'
                self.current_left_segment['coop_split'] = coop_split_step
            if self.current_right_segment:
                self.current_right_segment['top_arm'] = 'left'
                self.current_right_segment['coop_split'] = coop_split_step
            
            # Left stays home until end - REMOVED to allow async scheduling
            # self.add_wait(self.left_trajectory, right_return_t)

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
            
            # --- New Logic: Both Return Home -> Wait -> Base Place ---
            t_home_start = t_retreat
            t_home_arrival = t_home_start + 60
            
            # Both return home (Base keeps holding)
            self.right_trajectory.append({"t": t_home_arrival, "xyz": self.init_right_pose["xyz"], "quat": self.init_right_pose["quat"], "gripper": 1})
            self.left_trajectory.append({"t": t_home_arrival, "xyz": self.init_left_pose["xyz"], "quat": self.init_left_pose["quat"], "gripper": 0})

            # Wait 20 steps
            t_wait_end = t_home_arrival + 20
            self.add_wait(self.right_trajectory, t_wait_end) # Top Arm (Right) should NOT wait? No, Right is Top here... Wait.
            # If Right has Green (Top), Right is Top.
            # Base is Left.
            # Wait, line 372/381 block was "Left has Green (Top)". So Left shouldn't wait.
            
            # Block starting line 444 is "Else" (Right has Green = Top).
            # So Right (Top) shouldn't wait. Left (Base) should wait.
            
            # self.add_wait(self.right_trajectory, t_wait_end) # REMOVE WAIT
            self.add_wait(self.left_trajectory, t_wait_end) # Base waits

            # Place (Left carries Blue base)
            t_place_start = t_wait_end
            t_place_end = self.add_place(self.left_trajectory, t_place_start, np.array([0, 0.05, 0.025]), is_left=True)

            # Left returns home after placing
            left_return_t = t_place_end + 40
            self.left_trajectory.append({"t": left_return_t, "xyz": self.init_left_pose["xyz"], "quat": self.init_left_pose["quat"], "gripper": 1})
            
            # Record cooperative split point (middle of Top arm's 20-step wait at home)
            coop_split_step = t_home_arrival + 10
            if self.current_left_segment:
                self.current_left_segment['top_arm'] = 'right'
                self.current_left_segment['coop_split'] = coop_split_step
            if self.current_right_segment:
                self.current_right_segment['top_arm'] = 'right'
                self.current_right_segment['coop_split'] = coop_split_step
            
            # Right stays home until end - REMOVED
            # self.add_wait(self.right_trajectory, left_return_t)
        
        
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
    parser.add_argument('--color_sequence', type=str, help='Custom color sequence (e.g. "rrgbrrgbrr")', default=None)
    args = parser.parse_args()
    
    # Pre-populate command buffer
    initial_commands = list(args.commands)
    # Note: Logic moved to schedule_command/process loop
    
    # MANYCUBES_COLORS[0] is modified in place if arg provided, else uses default import
    if args.color_sequence:
        seq = list(args.color_sequence.lower())
        if len(seq) == 10 and all(c in ['r','g','b'] for c in seq):
            MANYCUBES_COLORS[0] = seq
            print(f"Using custom color sequence: {seq}")
        else:
            print("Invalid color sequence provided. Using default.")

    if not args.color_sequence:
        MANYCUBES_COLORS[0] = COLOR_SEQUENCE
    
    task_name = 'sim_many_cubes'
    env = make_ee_sim_env(task_name, camera_names=['top'])
    # Pass the (potentially modified) color sequence to the policy
    policy = InteractivePolicy(inject_noise=False, color_sequence=MANYCUBES_COLORS[0])
    
    # Fill policy buffer
    for c in initial_commands:
        policy.schedule_command(c)
    
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
    print(f"Sequence: {policy.color_sequence}")
    print("Commands:")
    print("  'I': Independent Pick (Splits into T then B)")
    print("  'C': Cooperative Assembly (Swaps after completion)")
    print("  'L'/'R': Explicit Left/Right Pick")
    print("  'T'/'B': Explicit Top/Base Role Pick")
    print("  'q': Quit")
    
    step = 0
    try:
        while True:
            # Process Buffer
            policy.process_command_buffer(ts)

            # Non-blocking stdin read
            if select.select([sys.stdin], [], [], 0.0)[0]:
                line = sys.stdin.readline().strip().upper()
                if line == 'Q':
                    break
                elif line in ['I', 'C', 'L', 'R']: # T and B are internal resolution, not direct user input
                    policy.schedule_command(line, ts)
                else:
                    print("Unknown command. Use I, C, L, R or Q.")
            
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
             
            # Deprecated: last_action_end_t check is less useful with async
            # if step == policy.last_action_end_t: ...

            step += 1
            
    except KeyboardInterrupt:
        pass
    finally:
        # plt.close()
        cv2.destroyAllWindows()
        print("Interactive Policy Terminated")

if __name__ == '__main__':
    main()
