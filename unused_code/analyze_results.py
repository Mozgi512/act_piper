
import csv
import math
import os
import statistics
import sys

import matplotlib.pyplot as plt


def _to_bool(value):
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return float(default)


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _save_summary_csv(out_dir, summary, per_obj):
    summary_path = os.path.join(out_dir, "summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in summary.items():
            writer.writerow([k, v])

    per_obj_path = os.path.join(out_dir, "per_object_stats.csv")
    with open(per_obj_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "object",
            "indep_success",
            "coop_success",
            "fail",
            "success_rate_percent",
        ])
        for i in range(10):
            obj = per_obj[i]
            total = obj["indep"] + obj["coop"] + obj["fail"]
            success_rate = 100.0 * (obj["indep"] + obj["coop"]) / total if total else 0.0
            writer.writerow([i, obj["indep"], obj["coop"], obj["fail"], f"{success_rate:.2f}"])


def _plot_reward_hist(out_dir, rewards):
    if not rewards:
        return
    bins = min(15, max(5, int(math.sqrt(len(rewards)))))
    plt.figure(figsize=(7, 4.5))
    plt.hist(rewards, bins=bins)
    plt.title("Total Reward Distribution")
    plt.xlabel("Total Reward")
    plt.ylabel("Episode Count")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "reward_hist.png"), dpi=160)
    plt.close()


def _plot_success_pie(out_dir, success_count, total_count):
    fail_count = max(0, total_count - success_count)
    plt.figure(figsize=(5.5, 5.5))
    plt.pie(
        [success_count, fail_count],
        labels=["Success", "Failure"],
        autopct="%.1f%%",
        startangle=90,
    )
    plt.title("Episode Success Ratio")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "episode_success_ratio.png"), dpi=160)
    plt.close()


def _plot_per_object_stacked(out_dir, per_obj):
    xs = list(range(10))
    indep = [per_obj[i]["indep"] for i in xs]
    coop = [per_obj[i]["coop"] for i in xs]
    fail = [per_obj[i]["fail"] for i in xs]

    plt.figure(figsize=(9, 4.8))
    plt.bar(xs, indep, label="Indep_Success")
    plt.bar(xs, coop, bottom=indep, label="Coop_Success")
    bottom_fail = [indep[i] + coop[i] for i in xs]
    plt.bar(xs, fail, bottom=bottom_fail, label="Fail")
    plt.xticks(xs, [f"Obj{i}" for i in xs])
    plt.ylabel("Count")
    plt.title("Per-Object Status Counts")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "per_object_status_stacked.png"), dpi=160)
    plt.close()


def analyze_csv(filepath, output_root=None):
    print(f"\nAnalyzing: {filepath}")
    if not os.path.exists(filepath):
        print("  File not found.")
        return None

    try:
        with open(filepath, "r") as f:
            reader = csv.DictReader(f)
            data = list(reader)
    except Exception as e:
        print(f"  Error reading CSV: {e}")
        return None

    num_episodes = len(data)
    if num_episodes == 0:
        print("  No data found.")
        return None

    success_count = 0
    rewards = []
    per_obj = {i: {"indep": 0, "coop": 0, "fail": 0} for i in range(10)}
    total_indep_success = 0
    total_coop_success = 0
    total_fail = 0

    for row in data:
        if _to_bool(row.get("Is_Success", False)):
            success_count += 1

        rew = _safe_float(row.get("Total_Reward", 0.0))
        rewards.append(rew)

        for i in range(10):
            status = str(row.get(f"Obj{i}_Status", "Fail")).strip()
            if status == "Indep_Success":
                per_obj[i]["indep"] += 1
                total_indep_success += 1
            elif status == "Coop_Success":
                per_obj[i]["coop"] += 1
                total_coop_success += 1
            else:
                per_obj[i]["fail"] += 1
                total_fail += 1

    success_rate = 100.0 * success_count / num_episodes
    avg_reward = statistics.mean(rewards)
    median_reward = statistics.median(rewards)
    std_reward = statistics.pstdev(rewards) if len(rewards) > 1 else 0.0

    summary = {
        "episodes": num_episodes,
        "episode_success_count": success_count,
        "episode_success_rate_percent": f"{success_rate:.2f}",
        "reward_mean": f"{avg_reward:.4f}",
        "reward_median": f"{median_reward:.4f}",
        "reward_std": f"{std_reward:.4f}",
        "reward_min": f"{min(rewards):.4f}",
        "reward_max": f"{max(rewards):.4f}",
        "object_indep_success_total": total_indep_success,
        "object_coop_success_total": total_coop_success,
        "object_fail_total": total_fail,
    }

    print(f"  Episodes: {num_episodes}")
    print(f"  Episode Success: {success_count}/{num_episodes} ({success_rate:.1f}%)")
    print(f"  Reward Mean/Median/Std: {avg_reward:.2f} / {median_reward:.2f} / {std_reward:.2f}")
    print(f"  Reward Min/Max: {min(rewards):.2f} / {max(rewards):.2f}")
    print(f"  Object Status Totals: Indep={total_indep_success}, Coop={total_coop_success}, Fail={total_fail}")

    print("  Per-Object Stats:")
    for i in range(10):
        obj = per_obj[i]
        obj_total = obj["indep"] + obj["coop"] + obj["fail"]
        obj_succ = obj["indep"] + obj["coop"]
        obj_succ_rate = 100.0 * obj_succ / obj_total if obj_total else 0.0
        print(
            f"    Obj{i}: {obj_succ_rate:5.1f}% "
            f"(Indep:{obj['indep']}, Coop:{obj['coop']}, Fail:{obj['fail']})"
        )

    csv_name = os.path.splitext(os.path.basename(filepath))[0]
    base_dir = output_root if output_root else os.path.dirname(filepath)
    out_dir = os.path.join(base_dir, f"{csv_name}_analysis")
    _ensure_dir(out_dir)

    _save_summary_csv(out_dir, summary, per_obj)
    _plot_reward_hist(out_dir, rewards)
    _plot_success_pie(out_dir, success_count, num_episodes)
    _plot_per_object_stacked(out_dir, per_obj)

    print(f"  Saved analysis to: {out_dir}")
    return {"summary": summary, "output_dir": out_dir}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 analyze_results.py <csv_file1> <csv_file2> ...")
        sys.exit(1)

    for f in sys.argv[1:]:
        analyze_csv(f)
