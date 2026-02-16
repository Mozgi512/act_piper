import argparse
import csv
import json
import os
import re
from collections import Counter, defaultdict
from statistics import mean, pstdev

import matplotlib.pyplot as plt


def parse_bool(text):
    return str(text).strip().lower() == "true"


def parse_float(text, default=0.0):
    try:
        return float(text)
    except Exception:
        return default


def parse_int(text, default=0):
    try:
        return int(float(text))
    except Exception:
        return default


def parse_concatenated_eval_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()

    header_key = "Obj9_Reward"
    header_end = text.find(header_key)
    if header_end == -1:
        raise ValueError(f"Could not find header terminator '{header_key}' in {file_path}")
    header_end += len(header_key)

    header = text[:header_end].strip()
    data_blob = text[header_end:].strip()

    columns = next(csv.reader([header]))
    expected_len = len(columns)

    row_texts = re.split(r"\s+(?=\d+,)", data_blob)
    rows = []
    for row_text in row_texts:
        row_text = row_text.strip()
        if not row_text:
            continue
        row = next(csv.reader([row_text]))
        if len(row) < expected_len:
            row.extend([""] * (expected_len - len(row)))
        elif len(row) > expected_len:
            row = row[:expected_len]
        rows.append(dict(zip(columns, row)))

    parsed = []
    for row in rows:
        rec = dict(row)
        rec["Episode"] = parse_int(row.get("Episode", 0), 0)
        rec["Total_Reward"] = parse_float(row.get("Total_Reward", 0), 0.0)
        rec["Is_Success"] = parse_bool(row.get("Is_Success", "False"))
        for idx in range(10):
            rec[f"Obj{idx}_Reward"] = parse_float(row.get(f"Obj{idx}_Reward", 0), 0.0)
        parsed.append(rec)

    parsed.sort(key=lambda r: r["Episode"])
    return parsed, columns


def summarize(records):
    rewards = [r["Total_Reward"] for r in records]
    success_count = sum(1 for r in records if r["Is_Success"])

    status_counter = Counter()
    color_status_counter = defaultdict(Counter)
    color_reward_values = defaultdict(list)

    for row in records:
        for idx in range(10):
            color = str(row.get(f"Obj{idx}_Color", "")).strip().lower()
            status = str(row.get(f"Obj{idx}_Status", "")).strip()
            reward = row.get(f"Obj{idx}_Reward", 0.0)
            if status:
                status_counter[status] += 1
            if color and status:
                color_status_counter[color][status] += 1
            if color in ["r", "g", "b"]:
                color_reward_values[color].append(float(reward))

    mean_reward_by_color = {
        c: (mean(vals) if vals else 0.0)
        for c, vals in color_reward_values.items()
    }

    return {
        "episodes": len(records),
        "success_count": success_count,
        "success_rate": (success_count / len(records)) if records else 0.0,
        "reward_mean": mean(rewards) if rewards else 0.0,
        "reward_std": pstdev(rewards) if len(rewards) >= 2 else 0.0,
        "reward_min": min(rewards) if rewards else 0.0,
        "reward_max": max(rewards) if rewards else 0.0,
        "status_counts": dict(status_counter),
        "color_status_counts": {k: dict(v) for k, v in color_status_counter.items()},
        "mean_object_reward_by_color": mean_reward_by_color,
    }


def save_parsed_csv(records, columns, out_path):
    with open(out_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in records:
            out_row = dict(row)
            out_row["Is_Success"] = "True" if row["Is_Success"] else "False"
            writer.writerow(out_row)


def make_plots(records, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    episodes = [r["Episode"] for r in records]
    total_rewards = [r["Total_Reward"] for r in records]

    plt.figure(figsize=(9, 4))
    plt.plot(episodes, total_rewards, marker="o", linewidth=1.8)
    success_eps = [r["Episode"] for r in records if r["Is_Success"]]
    success_rewards = [r["Total_Reward"] for r in records if r["Is_Success"]]
    if success_eps:
        plt.scatter(success_eps, success_rewards, marker="*", s=120, label="Success")
        plt.legend()
    plt.title("Total Reward by Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "reward_by_episode.png"), dpi=150)
    plt.close()

    status_counter = Counter()
    for row in records:
        for i in range(10):
            status = str(row.get(f"Obj{i}_Status", "")).strip()
            if status:
                status_counter[status] += 1

    labels = sorted(status_counter.keys())
    values = [status_counter[k] for k in labels]
    plt.figure(figsize=(7, 4))
    plt.bar(labels, values)
    plt.title("Object Status Counts (All Episodes)")
    plt.xlabel("Status")
    plt.ylabel("Count")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "status_counts.png"), dpi=150)
    plt.close()

    reward_by_color = defaultdict(list)
    for row in records:
        for i in range(10):
            color = str(row.get(f"Obj{i}_Color", "")).strip().lower()
            if color in ["r", "g", "b"]:
                reward_by_color[color].append(float(row.get(f"Obj{i}_Reward", 0.0)))

    color_order = ["r", "g", "b"]
    mean_rewards = [mean(reward_by_color[c]) if reward_by_color[c] else 0.0 for c in color_order]
    plt.figure(figsize=(6, 4))
    plt.bar(color_order, mean_rewards)
    plt.title("Mean Object Reward by Color")
    plt.xlabel("Color")
    plt.ylabel("Mean Reward")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "mean_reward_by_color.png"), dpi=150)
    plt.close()

    # Per-object stacked status counts (similar to results/E2E_random_analysis style)
    per_obj = {i: {"indep": 0, "coop": 0, "fail": 0} for i in range(10)}
    for row in records:
        for i in range(10):
            status = str(row.get(f"Obj{i}_Status", "")).strip()
            if status == "Indep_Success":
                per_obj[i]["indep"] += 1
            elif status == "Coop_Success":
                per_obj[i]["coop"] += 1
            elif status == "Fail":
                per_obj[i]["fail"] += 1

    xs = list(range(10))
    indep_vals = [per_obj[i]["indep"] for i in xs]
    coop_vals = [per_obj[i]["coop"] for i in xs]
    fail_vals = [per_obj[i]["fail"] for i in xs]
    fail_bottom = [indep_vals[i] + coop_vals[i] for i in xs]

    plt.figure(figsize=(9, 4.8))
    plt.bar(xs, indep_vals, label="Indep_Success")
    plt.bar(xs, coop_vals, bottom=indep_vals, label="Coop_Success")
    plt.bar(xs, fail_vals, bottom=fail_bottom, label="Fail")
    plt.xticks(xs, [f"Obj{i}" for i in xs])
    plt.ylabel("Count")
    plt.title("Per-Object Status Counts")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "per_object_status_stacked.png"), dpi=160)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="analysis_hie_results")
    args = parser.parse_args()

    records, columns = parse_concatenated_eval_file(args.input)
    summary = summarize(records)

    os.makedirs(args.out_dir, exist_ok=True)
    save_parsed_csv(records, columns, os.path.join(args.out_dir, "parsed_rows.csv"))
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    make_plots(records, args.out_dir)

    print("=== Summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved outputs to: {args.out_dir}")


if __name__ == "__main__":
    main()
