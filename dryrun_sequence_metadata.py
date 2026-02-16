import os
import csv
import json
import argparse
import numpy as np

from interactive_policy import InteractivePolicy
from piper_constants import SIM_TASK_CONFIGS
from piper_sim_env import MANYCUBES_COLORS, MANYCUBES_TASK_COUNT, MANYCUBES_CONFIG
from piper_ee_sim_env import make_ee_sim_env
from utils import set_seed


def is_valid_color_seq(s):
    s = s.strip().lower()
    return len(s) == 10 and all(c in ['r', 'g', 'b'] for c in s)


def is_success_label(s):
    return str(s).strip().lower() in {'success', 's', 'ok', '1', 'true'}


def load_sequence_rows(sequence_file, success_only=False):
    rows = []
    with open(sequence_file, 'r') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if not row:
                continue
            c0 = row[0].strip() if len(row) >= 1 else ''
            c1 = row[1].strip().upper() if len(row) >= 2 else ''
            c2 = row[2].strip() if len(row) >= 3 else ''
            if is_valid_color_seq(c0) and c1:
                if success_only and not is_success_label(c2):
                    continue
                rows.append({'row': i, 'color': c0.lower(), 'cmd': c1})
    return rows


def segments_to_pairs(left_segments, right_segments, num_steps):
    left = np.array(['H'] * num_steps, dtype=object)
    right = np.array(['H'] * num_steps, dtype=object)

    for seg in left_segments:
        s = int(seg.get('start', 0))
        e = int(seg.get('end', 0))
        s = max(0, min(s, num_steps))
        e = max(0, min(e, num_steps))
        if e <= s:
            continue
        typ = str(seg.get('type', '')).lower()
        mode = 'C' if typ == 'cooperative' else 'I' if typ == 'independent' else 'H'
        left[s:e] = mode

    for seg in right_segments:
        s = int(seg.get('start', 0))
        e = int(seg.get('end', 0))
        s = max(0, min(s, num_steps))
        e = max(0, min(e, num_steps))
        if e <= s:
            continue
        typ = str(seg.get('type', '')).lower()
        mode = 'C' if typ == 'cooperative' else 'I' if typ == 'independent' else 'H'
        right[s:e] = mode

    pair0 = f"{left[0]}{right[0]}" if num_steps > 0 else 'HH'
    transitions = []
    prev = pair0
    for t in range(1, num_steps):
        pair = f"{left[t]}{right[t]}"
        if pair != prev:
            transitions.append({'t': t, 'pair': pair})
            prev = pair

    return pair0, transitions


def main(args):
    set_seed(args.seed)

    rows = load_sequence_rows(args.sequence_file, success_only=args.success_only)
    if not rows:
        if args.success_only:
            raise RuntimeError(f'No valid success rows (color, command, status=success) in {args.sequence_file}')
        raise RuntimeError(f'No valid (color, command) rows in {args.sequence_file}')

    if args.shuffle:
        rng = np.random.default_rng(args.seed)
        rng.shuffle(rows)

    if args.num_episodes is not None:
        rows = rows[:args.num_episodes]

    task_cfg = SIM_TASK_CONFIGS[args.task_name]
    default_episode_len = int(task_cfg['episode_len'])
    max_steps = args.max_steps if args.max_steps is not None else max(3000, default_episode_len)

    out_rows = []

    for ep_idx, item in enumerate(rows):
        color_seq = item['color']
        cmd_seq = item['cmd']

        # Match generation settings for ManyCubes
        MANYCUBES_COLORS[0] = list(color_seq)
        MANYCUBES_TASK_COUNT[0] = len(cmd_seq)

        if args.random_x_shift:
            MANYCUBES_CONFIG['x_shift'] = float(np.random.uniform(0.00, 0.10))
        else:
            MANYCUBES_CONFIG['x_shift'] = float(args.x_shift)

        env = make_ee_sim_env(args.task_name, camera_names=[])
        ts = env.reset()

        policy = InteractivePolicy(inject_noise=False, color_sequence=list(color_seq))
        policy.init_pose(ts)
        for c in cmd_seq:
            policy.schedule_command(c, ts)

        final_step = 0
        for step in range(max_steps):
            policy.process_command_buffer(ts)
            if not policy.command_buffer:
                if policy.is_arm_free(True, step) and policy.is_arm_free(False, step):
                    final_step = step
                    break
            action = policy(ts)
            ts = env.step(action)
            final_step = step

        policy.finalize(final_step)

        left_segments = policy.left_segments.copy()
        right_segments = policy.right_segments.copy()
        num_steps = max(1, final_step + 1)
        pair0, transitions = segments_to_pairs(left_segments, right_segments, num_steps)

        out_rows.append({
            'episode': ep_idx,
            'sequence_row': item['row'],
            'color_sequence': color_seq,
            'command_sequence': cmd_seq,
            'num_steps': num_steps,
            'pair_at_t0': pair0,
            'num_transitions': len(transitions),
            'transitions_json': json.dumps(transitions, ensure_ascii=False),
            'left_segments_json': json.dumps(left_segments, ensure_ascii=False),
            'right_segments_json': json.dumps(right_segments, ensure_ascii=False),
        })

        del env

    os.makedirs(os.path.dirname(args.output_csv) or '.', exist_ok=True)
    with open(args.output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)

    print(f'Saved: {args.output_csv}')
    print(f'Episodes: {len(out_rows)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', type=str, default='sim_many_cubes')
    parser.add_argument('--sequence_file', type=str, required=True)
    parser.add_argument('--output_csv', type=str, required=True)
    parser.add_argument('--num_episodes', type=int, default=None)
    parser.add_argument('--max_steps', type=int, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--shuffle', action='store_true', default=True,
                        help='Randomize processing order (default: enabled)')
    parser.add_argument('--no_shuffle', action='store_false', dest='shuffle',
                        help='Disable randomization and keep CSV top-down order')
    parser.add_argument('--success_only', action='store_true',
                        help='Use only rows whose 3rd CSV column indicates success (success/s/ok/1/true)')
    parser.add_argument('--random_x_shift', action='store_true', help='Match generate_dataset random x-shift behavior')
    parser.add_argument('--x_shift', type=float, default=0.0)
    main(parser.parse_args())
