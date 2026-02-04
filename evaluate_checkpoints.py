import os
import subprocess
import re
import csv
import matplotlib.pyplot as plt
import pandas as pd
import multiprocessing
from functools import partial

# Configuration
CKPT_C_DIR = "ckpt/c"
CKPT_L_DIR = "ckpt/i_left"
CKPT_R_DIR = "ckpt/i_right"

# Base command arguments
BASE_CMD = [
    "python", "-u", "policy_switcher.py",
    "--task_name", "sim_many_cubes",
    "--policy_class", "ACT",
    "--kl_weight", "10",
    "--chunk_size", "100",
    "--hidden_dim", "512",
    "--dim_feedforward", "3200",
    "--command_sequence", "IIC",
    "--task_durations", '{"I": 380, "C": 520}',
    # "--onscreen_render", # Disabled for speed
    "--inherit_temporal_buffer",
    "--warmup_steps", "100",
    "--num_rollouts", "50",
    "--color_sequence", "rrrrgbgbrr",
    "--disable_hold"
]

def run_single_evaluation(label, ckpt_filename):
    """
    Worker function to evaluate a single checkpoint.
    Returns a dictionary with results or None if failed.
    """
    ckpt_c = os.path.join(CKPT_C_DIR, ckpt_filename)
    ckpt_l = os.path.join(CKPT_L_DIR, ckpt_filename)
    ckpt_r = os.path.join(CKPT_R_DIR, ckpt_filename)
    
    # Check if files exist
    if not (os.path.exists(ckpt_c) and os.path.exists(ckpt_l) and os.path.exists(ckpt_r)):
        print(f"Skipping {label}: Checkpoint(s) missing.")
        return None

    print(f"Starting evaluation for {label}...")
    
    cmd = BASE_CMD + [
        "--ckpt_dual", ckpt_c,
        "--ckpt_left", ckpt_l,
        "--ckpt_right", ckpt_r
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
        
        # Read output line by line (captured but not printed to avoid interleaving)
        for line in iter(process.stdout.readline, ''):
            output_lines.append(line)
            # Parse per-episode MaxReward
            # Line format: "Episode {count}: Return={ret}, MaxReward={max_r}"
            ep_match = re.search(r"Episode (\d+): Return=[\d\.]+, MaxReward=([\d\.]+)", line)
            if ep_match:
                ep_num = int(ep_match.group(1))
                epoch_max_rewards.append(float(ep_match.group(2)))
                
                # Report progress every 10 episodes
                if (ep_num + 1) % 10 == 0:
                    print(f"[{label}] Progress: {ep_num + 1}/50")
            
        process.stdout.close()
        return_code = process.wait()
        
        full_output = "".join(output_lines)
        
        if return_code != 0:
            print(f"Error evaluating {label} (Exit Code {return_code})")
            return None
            
        # Parse Success Rate
        success_match = re.search(r"Success Rate \(MaxReward=4\): ([\d\.]+)%", full_output)
        return_match = re.search(r"Average Return: ([\d\.]+)", full_output)
        
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

def evaluate_checkpoints():
    start_epoch = 1000
    end_epoch = 39000
    step_size = 1000
    num_workers = 3
    
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
    
    print(f"Starting parallel evaluation with {num_workers} workers...")
    
    with multiprocessing.Pool(processes=num_workers) as pool:
        # Use imap_unordered to get results as they finish
        for result in pool.imap_unordered(run_single_evaluation_wrapper, tasks):
            if result:
                results.append(result)
                # Incremental Save
                df = pd.DataFrame(results)
                # Sort by epoch for readability in CSV (optional but nice)
                # Note: 'best' will be at bottom or top depending on sort, handle mixed types
                
                # Simple sort helper
                def sort_key(x):
                    e = x['Epoch']
                    return int(e) if isinstance(e, int) or (isinstance(e, str) and e.isdigit()) else 999999
                
                results_sorted = sorted(results, key=sort_key)
                pd.DataFrame(results_sorted).to_csv("evaluation_results.csv", index=False)
                print(f"Saved results ({len(results)}/{len(tasks)}) to evaluation_results.csv")
    
    # Final Plotting
    if results:
        df = pd.DataFrame(results)
        
        # Filter numeric epochs for line plot
        df_numeric = df[df['Epoch'] != 'best'].copy()
        df_numeric['Epoch'] = pd.to_numeric(df_numeric['Epoch'])
        df_numeric = df_numeric.sort_values('Epoch')
        
        plt.figure(figsize=(10, 6))
        plt.plot(df_numeric['Epoch'], df_numeric['SuccessRate'], marker='o', linestyle='-', color='b', label='Epochs')
        
        # Handle Best
        best_row = df[df['Epoch'] == 'best']
        if not best_row.empty:
            best_val = best_row.iloc[0]['SuccessRate']
            plt.axhline(y=best_val, color='r', linestyle='--', label=f'Best ({best_val}%)')
            
        plt.title('Success Rate Evolution')
        plt.xlabel('Epoch')
        plt.ylabel('Success Rate (%)')
        plt.grid(True)
        plt.legend()
        plt.savefig('success_rate_evolution.png')
        print("Plot saved to success_rate_evolution.png")
    else:
        print("No results to plot.")

if __name__ == "__main__":
    evaluate_checkpoints()
