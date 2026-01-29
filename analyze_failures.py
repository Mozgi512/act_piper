
import os
import ast
import numpy as np
import glob

def analyze_checkpoint(ckpt_dir, max_possible_reward=4):
    result_path = os.path.join(ckpt_dir, 'result_policy_best.txt')
    if not os.path.exists(result_path):
        # analyze_failures.py
        print(f"Skipping {ckpt_dir} (No result file)")
        return

    with open(result_path, 'r') as f:
        lines = f.readlines()
        
    # Find the line with the highest rewards list. 
    # Based on file inspection, it's the last non-empty line usually, or explicitly the one after returns.
    # The file has: returns on line 10, rewards on line 12.
    # Let's verify by checking if it looks like a list.
    
    highest_rewards = None
    for line in reversed(lines):
        line = line.strip()
        if line.startswith('[') and line.endswith(']'):
            try:
                data = ast.literal_eval(line)
                # Heuristic: Rewards list usually has small integers (0-4), Returns list has large sums (1000+)
                if np.max(data) <= max_possible_reward:
                    highest_rewards = np.array(data)
                    break
            except:
                continue
                
    if highest_rewards is None:
        print(f"Could not find rewards list in {result_path}")
        return

    failed_rewards = highest_rewards[highest_rewards < max_possible_reward]
    
    print(f"--- {ckpt_dir} ---")
    print(f"Total Episodes: {len(highest_rewards)}")
    print(f"Failed Episodes: {len(failed_rewards)} ({len(failed_rewards)/len(highest_rewards)*100:.1f}%)")
    
    if len(failed_rewards) > 0:
        print(f"Failed Max Reward Stats:")
        print(f"  Mean: {np.mean(failed_rewards):.4f}")
        print(f"  Std : {np.std(failed_rewards):.4f}")
        print(f"  Min : {np.min(failed_rewards)}")
        print(f"  Max : {np.max(failed_rewards)}")
        
        # Distribution
        unique, counts = np.unique(failed_rewards, return_counts=True)
        print(f"  Distribution: {dict(zip(unique, counts))}")
    else:
        print("  No failures.")
    print("")

def main():
    dirs = [
        'ckpt/cooperation_50eps',
        'ckpt/cooperation_75eps',
        'ckpt/cooperation',       # 100 eps
        'ckpt/cooperation_150eps'
    ]
    
    for d in dirs:
        analyze_checkpoint(d)

if __name__ == '__main__':
    main()
