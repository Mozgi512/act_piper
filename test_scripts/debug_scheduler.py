
from policy_switcher import TaskScheduler

command_sequence = "ICRICL"
durations = {"I": 380, "C_assembly": 400, "C_place": 120, "Single": 380}

print(f"Testing Scheduler with: {command_sequence}")
print(f"Durations: {durations}")

scheduler = TaskScheduler(command_sequence, durations)
scheduler.print_schedule()

print("Check t=10 states:")
print(f"Left: {scheduler.get_arm_state(10, 'left')}")
print(f"Right: {scheduler.get_arm_state(10, 'right')}")
