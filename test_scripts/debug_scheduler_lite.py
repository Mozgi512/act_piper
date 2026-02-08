
import json

MODE_INDEPENDENT = '1'
MODE_COOP = '2'

class TaskScheduler:
    def __init__(self, sequence_str, duration_config):
        self.sequence_str = sequence_str
        self.config = duration_config
        self.mode_schedule = {} # step -> mode
        self.hold_schedule = {'left': [], 'right': []} 
        
        self.timeline_left = [] # List of {'start':, 'end':, 'type':, 'info':}
        self.timeline_right = []
        
        self.max_timesteps = 0
        
        self.MODE_INDEPENDENT = MODE_INDEPENDENT
        self.MODE_COOP = MODE_COOP
        
        self._calculate_schedule()

    def _calculate_schedule(self):
        time_l = 0
        time_r = 0
        
        n = len(self.sequence_str)
        for i in range(n):
            char = self.sequence_str[i]
            
            if char == 'I':
                dur = self.config.get('I', 0)
                if dur == 0: print(f"WARNING: Duration for 'I' is 0!")
                # Independent Parallel
                start_l = time_l
                end_l = start_l + dur
                self.timeline_left.append({'start': start_l, 'end': end_l, 'type': 'INDEP', 'info': 'Parallel I'})
                time_l = end_l
                
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
                
                # Schedule Holds
                if time_l < start_coop:
                    self.timeline_left.append({'start': time_l, 'end': start_coop, 'type': 'HOLD', 'info': 'Sync Wait'})
                    time_l = start_coop
                
                if time_r < start_coop:
                    self.timeline_right.append({'start': time_r, 'end': start_coop, 'type': 'HOLD', 'info': 'Sync Wait'})
                    time_r = start_coop
                
                # Mode Switch Registration
                self.mode_schedule[start_coop] = self.MODE_COOP
                
                # Phase 1: Assembly (Coop Mode)
                end_assembly = start_coop + len_assembly
                self.timeline_left.append({'start': start_coop, 'end': end_assembly, 'type': 'COOP', 'info': 'Phase 1 (Assembly)'})
                self.timeline_right.append({'start': start_coop, 'end': end_assembly, 'type': 'COOP', 'info': 'Phase 1 (Assembly)'})
                
                # Phase 2: Placement (Indep Mode)
                self.mode_schedule[end_assembly] = self.MODE_INDEPENDENT
                
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
                
                end_place = end_assembly + len_place
                
                if base_arm == 'left':
                    # Left blocked (Base), Right free (Top)
                    self.timeline_left.append({'start': end_assembly, 'end': end_place, 'type': 'INDEP', 'info': 'Phase 2 (Place-Base)'})
                    time_l = end_place
                    time_r = end_assembly # Right free immediately
                else:
                    # Right blocked (Base), Left free (Top)
                    self.timeline_right.append({'start': end_assembly, 'end': end_place, 'type': 'INDEP', 'info': 'Phase 2 (Place-Base)'})
                    time_r = end_place
                    time_l = end_assembly # Left free immediately
                
        self.max_timesteps = max(time_l, time_r)
        
    def print_schedule(self):
        print("\n=== Left Arm Schedule ===")
        print(f"{'Start':<6} | {'End':<6} | {'Type':<10} | {'Info'}")
        for item in self.timeline_left:
            print(f"{item['start']:<6} | {item['end']:<6} | {item['type']:<10} | {item['info']}")
        print("\n=== Right Arm Schedule ===")
        print(f"{'Start':<6} | {'End':<6} | {'Type':<10} | {'Info'}")
        for item in self.timeline_right:
            print(f"{item['start']:<6} | {item['end']:<6} | {item['type']:<10} | {item['info']}")
        print("=======================\n")

    def get_arm_state(self, t, arm):
        timeline = self.timeline_left if arm == 'left' else self.timeline_right
        for item in timeline:
            if item['start'] <= t < item['end']:
                return item['type'], item['info']
        return 'HOLD', 'Idle/Finished'

command_sequence = "ICRICL"
durations = {"I": 380, "C_assembly": 400, "C_place": 120, "Single": 380}

print(f"Testing Scheduler with: {command_sequence}")
print(f"Durations: {durations}")

scheduler = TaskScheduler(command_sequence, durations)
scheduler.print_schedule()

print("-" * 20)
print(f"Time 0 Left: {scheduler.get_arm_state(0, 'left')}")
print(f"Time 0 Right: {scheduler.get_arm_state(0, 'right')}")
print(f"Time 10 Left: {scheduler.get_arm_state(10, 'left')}")
print(f"Time 10 Right: {scheduler.get_arm_state(10, 'right')}")
