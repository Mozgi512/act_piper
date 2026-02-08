import os
import subprocess
import re
import csv
import matplotlib.pyplot as plt
import pandas as pd
import multiprocessing
from functools import partial

# Configuration
CKPT_DIR = "ckpt/multitask_async"

# Base command arguments for evaluate_policy_switcher.py
# Based on USAGE.txt
BASE_CMD = [
    "python", "-u", "evaluate_policy_switcher.py",
    "--task_name", "sim_many_cubes",
    "--commands", "IIC",
    "--color_sequence", "rrgbrrrbgr",
    "--num_rollouts", "50",
    "--policy_class", "ACT",
    "--kl_weight", "10",
    "--chunk_size", "100",
    "--hidden_dim", "512",
    "--dim_feedforward", "3200",
    # "--onscreen_render", # Disabled for speed
    # "--save_video",      # Disabled for speed
    "--max_timesteps", "1300"
]

def run_single_evaluation(label, ckpt_filename):
    """
    Worker function to evaluate a single checkpoint.
    """
    ckpt_path = os.path.join(CKPT_DIR, ckpt_filename)
    
    # Check if files exist
    if not os.path.exists(ckpt_path):
        print(f"Skipping {label}: Checkpoint missing ({ckpt_path}).")
        return None

    print(f"Starting evaluation for {label}...")
    
    cmd = BASE_CMD + [
        "--ckpt_e2e", ckpt_path
    ]
    
    try:
        # Run evaluation (blocking call for worker)
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, 
            text=True, 
            bufsize=1
        )
        
        output_lines = []
        epoch_max_rewards = []
        
        # Read output line by line
        for line in iter(process.stdout.readline, ''):
            output_lines.append(line)
            
            # Parse per-episode Reward (from evaluate_policy_switcher.py)
            # Format: "Episode {count} Finished. Reward: {current_reward}/{max_possible_reward}. Success: {is_success}"
            ep_match = re.search(r"Episode (\d+) Finished\. Reward: ([\d\.]+)/[\d\.]+\. Success:", line)
            if ep_match:
                ep_num = int(ep_match.group(1))
                reward = float(ep_match.group(2))
                epoch_max_rewards.append(reward)
                
                # Report progress every 10 episodes
                if (ep_num + 1) % 10 == 0:
                     print(f"[{label}] Progress: {ep_num + 1}/50")
            
        process.stdout.close()
        return_code = process.wait()
        
        full_output = "".join(output_lines)
        
        if return_code != 0:
            print(f"Error evaluating {label} (Exit Code {return_code})")
            # print(full_output) # Optional: print full output on error
            return None
            
        # Parse Success Rate (from evaluate_policy_switcher.py)
        # Format: "Success Rate:   {rate}% ({count}/{total})"
        success_match = re.search(r"Success Rate:\s+([\d\.]+)%", full_output)
        
        # Parse Average Return
        # Format: "Average Reward: {avg}"
        return_match = re.search(r"Average Reward: ([\d\.]+)", full_output)
        
        if success_match:
            success_rate = float(success_match.group(1))
            avg_return = float(return_match.group(1)) if return_match else 0.0
            
            print(f"Finished {label}: Success Rate = {success_rate}%")
            
            return {
                "Epoch": label,
                "SuccessRate": success_rate,
                "AverageReturn": avg_return,
                "AllMaxRewards": str(epoch_max_rewards)
            }
        else:
            print(f"Could not parse success rate for {label}.")
            return None
            
    except Exception as e:
        print(f"Exception during evaluation of {label}: {e}")
        return None

def run_single_evaluation_wrapper(args):
    """Wrapper to unpack arguments for map functions."""
    return run_single_evaluation(*args)

def evaluate_e2e_checkpoints():
    # E2E Checkpoints often go up to ~37000 or similar
    start_epoch = 1000
    end_epoch = 37000 # Adjusted based on file list
    step_size = 1000
    num_workers = 3 # Safe limit for GPU
    
    tasks = []
    
    # Generate all epochs
    all_epochs = list(range(start_epoch, end_epoch + 1, step_size))
    
    # Create "Scattered" order (Breadth-First Decomposition)
    ordered_indices = []
    queue = [(0, len(all_epochs) - 1)]
    
    if len(all_epochs) > 0:
        ordered_indices.append(len(all_epochs) - 1)
        if len(all_epochs) > 1:
            ordered_indices.append(0)
    
    while queue:
        low, high = queue.pop(0)
        if high - low <= 1:
            continue
        mid = (low + high) // 2
        if mid not in ordered_indices:
            ordered_indices.append(mid)
        queue.append((low, mid))
        queue.append((mid, high))
        
    seen_epochs = set()
    for idx in ordered_indices:
        epoch = all_epochs[idx]
        if epoch not in seen_epochs:
            tasks.append((epoch, f"policy_epoch_{epoch}_seed_0.ckpt"))
            seen_epochs.add(epoch)
            
    tasks.append(("best", "policy_best.ckpt"))
    
    results = []
    
    print(f"Starting parallel E2E evaluation with {num_workers} workers...")
    
    with multiprocessing.Pool(processes=num_workers) as pool:
        # Use imap_unordered to get results as they finish
        for result in pool.imap_unordered(run_single_evaluation_wrapper, tasks):
            if result:
                results.append(result)
                # Incremental Save
                df = pd.DataFrame(results)
                
                # Simple sort helper
                def sort_key(x):
                    e = x['Epoch']
                    return int(e) if isinstance(e, int) or (isinstance(e, str) and e.isdigit()) else 999999
                
                results_sorted = sorted(results, key=sort_key)
                pd.DataFrame(results_sorted).to_csv("evaluation_results_e2e.csv", index=False)
                print(f"Saved results ({len(results)}/{len(tasks)}) to evaluation_results_e2e.csv")
    
    # Final Plotting
    if results:
        df = pd.DataFrame(results)
        
        # Filter numeric epochs for line plot
        df_numeric = df[df['Epoch'] != 'best'].copy()
        df_numeric['Epoch'] = pd.to_numeric(df_numeric['Epoch'])
        df_numeric = df_numeric.sort_values('Epoch')
        
        plt.figure(figsize=(10, 6))
        plt.plot(df_numeric['Epoch'], df_numeric['SuccessRate'], marker='o', linestyle='-', color='g', label='E2E Epochs')
        
        # Handle Best
        best_row = df[df['Epoch'] == 'best']
        if not best_row.empty:
            best_val = best_row.iloc[0]['SuccessRate']
            plt.axhline(y=best_val, color='r', linestyle='--', label=f'Best ({best_val}%)')
            
        plt.title('Success Rate Evolution (E2E)')
        plt.xlabel('Epoch')
        plt.ylabel('Success Rate (%)')
        plt.grid(True)
        plt.legend()
        plt.savefig('success_rate_evolution_e2e.png')
        print("Plot saved to success_rate_evolution_e2e.png")
    else:
        print("No results to plot.")

if __name__ == "__main__":
    evaluate_e2e_checkpoints()
