
import csv
import sys
import os

def analyze_csv(filepath):
    print(f"\nAnalyzing: {filepath}")
    if not os.path.exists(filepath):
        print("  File not found.")
        return

    try:
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            data = list(reader)
    except Exception as e:
        print(f"  Error reading CSV: {e}")
        return

    num_episodes = len(data)
    if num_episodes == 0:
        print("  No data found.")
        return

    total_success = 0
    total_reward = 0.0
    
    obj_stats = {}
    for i in range(10):
        obj_stats[i] = {'success': 0, 'indep': 0, 'coop': 0, 'fail': 0}

    for row in data:
        # Success
        is_success_str = row.get('Is_Success', 'False')
        if is_success_str.lower() == 'true':
            total_success += 1
        
        # Reward
        reward_str = row.get('Total_Reward', '0')
        try:
            total_reward += float(reward_str)
        except ValueError:
            pass

        # Per Object
        for i in range(10):
            status = row.get(f'Obj{i}_Status', 'Fail')
            if status == 'Indep_Success':
                obj_stats[i]['indep'] += 1
                obj_stats[i]['success'] += 1
            elif status == 'Coop_Success':
                obj_stats[i]['coop'] += 1
                obj_stats[i]['success'] += 1
            else:
                obj_stats[i]['fail'] += 1

    success_rate = (total_success / num_episodes) * 100
    avg_reward = total_reward / num_episodes

    print(f"  Episodes: {num_episodes}")
    print(f"  Success Rate: {success_rate:.1f}%")
    print(f"  Avg Reward:   {avg_reward:.2f}")

    print("  Per-Object Stats:")
    for i in range(10):
        s = obj_stats[i]
        success_pct = (s['success'] / num_episodes) * 100 if num_episodes > 0 else 0
        print(f"    Obj{i}: {success_pct:5.1f}% (Indep:{s['indep']}, Coop:{s['coop']}, Fail:{s['fail']})")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 analyze_results.py <csv_file1> <csv_file2> ...")
        sys.exit(1)
        
    for f in sys.argv[1:]:
        analyze_csv(f)
