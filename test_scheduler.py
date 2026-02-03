
import argparse

import argparse

class TaskScheduler:
    def __init__(self, sequence_str, duration_config):
        self.sequence_str = sequence_str
        self.config = duration_config
        self.mode_schedule = {} # step -> mode
        self.hold_schedule = {'left': [], 'right': []} 
        
        # New: Detailed timelines
        self.timeline_left = [] # List of {'start':, 'end':, 'type':, 'info':}
        self.timeline_right = []
        
        self.max_timesteps = 0
        
        self.MODE_INDEPENDENT = 'INDEPENDENT'
        self.MODE_COOP = 'COOP'
        
        self._calculate_schedule()

    def _calculate_schedule(self):
        # Current 'free' time for each arm
        time_l = 0
        time_r = 0
        
        n = len(self.sequence_str)
        for i in range(n):
            char = self.sequence_str[i]
            
            if char == 'I':
                dur = self.config.get('I', 0)
                
                # Independent Parallel
                # Left
                start_l = time_l
                end_l = start_l + dur
                self.timeline_left.append({'start': start_l, 'end': end_l, 'type': 'INDEP', 'info': 'Parallel I'})
                time_l = end_l
                
                # Right
                start_r = time_r
                end_r = start_r + dur
                self.timeline_right.append({'start': start_r, 'end': end_r, 'type': 'INDEP', 'info': 'Parallel I'})
                time_r = end_r
                
            elif char == 'L':
                dur = self.config.get('Single', 0)
                # Left Only
                start_l = time_l
                end_l = start_l + dur
                self.timeline_left.append({'start': start_l, 'end': end_l, 'type': 'INDEP', 'info': 'Single L'})
                time_l = end_l
                
            elif char == 'R':
                dur = self.config.get('Single', 0)
                # Right Only
                start_r = time_r
                end_r = start_r + dur
                self.timeline_right.append({'start': start_r, 'end': end_r, 'type': 'INDEP', 'info': 'Single R'})
                time_r = end_r

            elif char == 'C':
                len_assembly = self.config.get('C_assembly', 0)
                len_place = self.config.get('C_place', 0)
                
                # Sync Point
                start_coop = max(time_l, time_r)
                
                # Schedule Holds (Space filling)
                if time_l < start_coop:
                    self.hold_schedule['left'].append((time_l, start_coop))
                    self.timeline_left.append({'start': time_l, 'end': start_coop, 'type': 'HOLD', 'info': 'Sync Wait'})
                    time_l = start_coop
                
                if time_r < start_coop:
                    self.hold_schedule['right'].append((time_r, start_coop))
                    self.timeline_right.append({'start': time_r, 'end': start_coop, 'type': 'HOLD', 'info': 'Sync Wait'})
                    time_r = start_coop
                
                # Mode Switch Registration
                self.mode_schedule[start_coop] = self.MODE_COOP
                
                # Phase 1: Assembly (Combined)
                end_assembly = start_coop + len_assembly
                self.timeline_left.append({'start': start_coop, 'end': end_assembly, 'type': 'COOP', 'info': 'Phase 1 (Assembly)'})
                self.timeline_right.append({'start': start_coop, 'end': end_assembly, 'type': 'COOP', 'info': 'Phase 1 (Assembly)'})
                
                # Phase 2: Placement (Base stays COOP, Free becomes INDEP)
                end_place = end_assembly + len_place
                
                # Global Mode remains COOP until end_place
                self.mode_schedule[end_assembly] = self.MODE_COOP
                self.mode_schedule[end_place] = self.MODE_INDEPENDENT
                
                # Lookahead for Role Assignment
                base_arm = 'right' # Default
                if i + 1 < n:
                    next_char = self.sequence_str[i+1]
                    if next_char == 'L':
                        base_arm = 'right'
                    elif next_char == 'R':
                        base_arm = 'left'
                    else:
                        base_arm = 'right'
                
                if base_arm == 'left':
                    # Left blocked (Base) -> Continues COOP
                    self.timeline_left.append({'start': end_assembly, 'end': end_place, 'type': 'COOP', 'info': 'Phase 2 (Place-Base)'})
                    time_l = end_place
                    time_r = end_assembly # Right free immediately
                else:
                    # Right blocked (Base) -> Continues COOP
                    self.timeline_right.append({'start': end_assembly, 'end': end_place, 'type': 'COOP', 'info': 'Phase 2 (Place-Base)'})
                    time_r = end_place
                    time_l = end_assembly # Left free immediately
                
        self.max_timesteps = max(time_l, time_r)

    def print_arm_schedules(self):
        print("\n=== Left Arm Schedule ===")
        print(f"{'Start':<6} | {'End':<6} | {'Type':<10} | {'Info'}")
        print("-" * 40)
        for item in self.timeline_left:
            print(f"{item['start']:<6} | {item['end']:<6} | {item['type']:<10} | {item['info']}")
            
        print("\n=== Right Arm Schedule ===")
        print(f"{'Start':<6} | {'End':<6} | {'Type':<10} | {'Info'}")
        print("-" * 40)
        for item in self.timeline_right:
            print(f"{item['start']:<6} | {item['end']:<6} | {item['type']:<10} | {item['info']}")

        print("\n=== Mode Switches (Global) ===")
        sorted_keys = sorted(self.mode_schedule.keys())
        for t in sorted_keys:
            print(f"Step {t:<6}: Switch to {self.mode_schedule[t]}")

def main():
    parser = argparse.ArgumentParser(description="Test TaskScheduler Logic")
    parser.add_argument('sequence', nargs='?', default="ICLR", help="Task sequence string (e.g. 'ICLR', 'ICI')")
    args = parser.parse_args()
    
    sequence = args.sequence
    
    # Standard Durations (Modify here if needed)
    config = {
        'I': 380,          # Independent (Parallel L+R)
        'C_assembly': 400, # Coop Phase 1
        'C_place': 120,    # Coop Phase 2
        'Single': 380      # Single Arm (L or R)
    }
    
    print(f"\nRunning Scheduler Test for Sequence: {sequence}")
    print(f"Durations: {config}")
    
    scheduler = TaskScheduler(sequence, config)
    scheduler.print_arm_schedules()


if __name__ == "__main__":
    main()
