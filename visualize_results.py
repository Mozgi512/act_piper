
import csv
import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

def parse_csv(filepath):
    """
    Parses a results CSV and returns metrics.
    """
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return None

    try:
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            data = list(reader)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

    if not data:
        return None

    # Calculate metrics
    num_episodes = len(data)
    if num_episodes == 0: return None

    total_success = 0
    total_reward = 0.0
    
    # helper for boolean parsing
    def to_bool(s):
        return s.lower() == 'true'

    # Per-Object Stats: {0: {'success': N, 'total': N}, ...}
    obj_stats = {i: 0 for i in range(10)}
    
    for row in data:
        # Overall Success
        if to_bool(row.get('Is_Success', 'False')):
            total_success += 1
        
        # Total Reward
        try:
            total_reward += float(row.get('Total_Reward', '0'))
        except: pass

        # Per Object (Obj0_Status... Obj9_Status)
        for i in range(10):
            # Check for success status (Indep_Success or Coop_Success)
            status = row.get(f'Obj{i}_Status', 'Fail')
            if status in ['Indep_Success', 'Coop_Success']:
                obj_stats[i] += 1
    
    metrics = {
        'name': os.path.splitext(os.path.basename(filepath))[0],
        'episodes': num_episodes,
        'success_rate': (total_success / num_episodes) * 100,
        'avg_reward': total_reward / num_episodes,
        'obj_success_rates': [(obj_stats[i] / num_episodes) * 100 for i in range(10)]
    }
    return metrics

def visualize(files, output_file):
    results = []
    for f in files:
        m = parse_csv(f)
        if m: results.append(m)
    
    if not results:
        print("No valid data to visualize.")
        return

    # Names for legend/labels
    names = [r['name'] for r in results]
    
    # 1. Overall Success Rate Comparison
    success_rates = [r['success_rate'] for r in results]
    avg_rewards = [r['avg_reward'] for r in results]
    
    # Setup Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Bar Chart: Success Rate
    x = np.arange(len(names))
    width = 0.35
    
    bars1 = axes[0].bar(x, success_rates, width, color='skyblue', label='Success Rate (%)')
    axes[0].set_ylabel('Success Rate (%)')
    axes[0].set_title('Overall Task Success Rate')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names)
    axes[0].set_ylim(0, 105)
    
    # Add labels
    for bar in bars1:
        height = bar.get_height()
        axes[0].annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    # Bar Chart: Avg Reward
    # If rewards are large, use log scale or just raw
    bars2 = axes[1].bar(x, avg_rewards, width, color='lightgreen', label='Avg Reward')
    axes[1].set_ylabel('Average Reward')
    axes[1].set_title('Average Total Reward')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names)
    
    for bar in bars2:
        height = bar.get_height()
        axes[1].annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(output_file + "_overall.png")
    print(f"Saved {output_file}_overall.png")
    
    # 2. Per-Object Success Rate (Line Chart)
    plt.figure(figsize=(12, 6))
    
    objects = list(range(10))
    markers = ['o', 's', '^', 'D', 'v', '<', '>']
    
    for i, res in enumerate(results):
        plt.plot(objects, res['obj_success_rates'], marker=markers[i % len(markers)], label=res['name'], linewidth=2)
    
    plt.xlabel('Object Index (0-9)')
    plt.ylabel('Success Rate (%)')
    plt.title('Per-Object Success Rates')
    plt.xticks(objects, [f'Obj{i}' for i in objects])
    plt.ylim(-5, 105)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_file + "_objects.png")
    print(f"Saved {output_file}_objects.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize experiment results.")
    parser.add_argument('files', nargs='+', help='CSV files to compare')
    parser.add_argument('--out', default='results_comparison', help='Output filename prefix')
    args = parser.parse_args()
    
    visualize(args.files, args.out)
