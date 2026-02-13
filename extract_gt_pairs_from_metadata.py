import os
import re
import csv
import json
import glob
import argparse
import h5py
import numpy as np


def parse_episode_index(path):
    name = os.path.basename(path)
    m = re.search(r'episode_(\d+)\.hdf5$', name)
    return int(m.group(1)) if m else -1


def decode_text(value):
    if isinstance(value, bytes):
        return value.decode('utf-8', errors='ignore')
    if isinstance(value, np.bytes_):
        return value.astype(str)
    return str(value)


def seg_type_to_mode(seg_type):
    t = decode_text(seg_type).strip().lower()
    if t == 'independent':
        return 'I'
    if t == 'cooperative':
        return 'C'
    return 'H'


def load_sequence_file(sequence_file):
    seqs = []
    if not sequence_file or not os.path.exists(sequence_file):
        return seqs

    def is_valid_seq(s):
        s = s.strip().lower()
        return len(s) == 10 and all(ch in ['r', 'g', 'b'] for ch in s)

    with open(sequence_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            c0 = row[0].strip() if len(row) >= 1 else ''
            c1 = row[1].strip() if len(row) >= 2 else ''
            if is_valid_seq(c0):
                seqs.append(c0.lower())
            elif is_valid_seq(c1):
                seqs.append(c1.lower())
    return seqs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--output_csv', type=str, default='gt_mode_pairs.csv')
    parser.add_argument('--sequence_file', type=str, default=None,
                        help='Optional sequence CSV. Used only with --assume_sequence_by_episode_index.')
    parser.add_argument('--assume_sequence_by_episode_index', action='store_true',
                        help='Assume episode_i uses i-th valid sequence row (only if you know this is true).')
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.dataset_dir, 'episode_*.hdf5')), key=parse_episode_index)
    if not files:
        raise FileNotFoundError(f'No episode_*.hdf5 found in {args.dataset_dir}')

    seqs = load_sequence_file(args.sequence_file)

    rows = []
    for fp in files:
        ep = parse_episode_index(fp)
        with h5py.File(fp, 'r') as f:
            num_steps = int(f['/action'].shape[0]) if '/action' in f else int(f['/observations/qpos'].shape[0])

            left = np.array(['H'] * num_steps, dtype=object)
            right = np.array(['H'] * num_steps, dtype=object)

            if '/metadata/left_segments' in f:
                l_segs = f['/metadata/left_segments'][()]
                for seg in l_segs:
                    s = int(seg['start'])
                    e = int(seg['end'])
                    s = max(0, min(s, num_steps))
                    e = max(0, min(e, num_steps))
                    if e > s:
                        left[s:e] = seg_type_to_mode(seg['type'])

            if '/metadata/right_segments' in f:
                r_segs = f['/metadata/right_segments'][()]
                for seg in r_segs:
                    s = int(seg['start'])
                    e = int(seg['end'])
                    s = max(0, min(s, num_steps))
                    e = max(0, min(e, num_steps))
                    if e > s:
                        right[s:e] = seg_type_to_mode(seg['type'])

            transitions = []
            prev_pair = f"{left[0]}{right[0]}" if num_steps > 0 else 'HH'
            for t in range(1, num_steps):
                pair = f"{left[t]}{right[t]}"
                if pair != prev_pair:
                    transitions.append({'t': t, 'pair': pair})
                    prev_pair = pair

            color_seq = ''
            color_source = 'missing_in_hdf5'
            if '/metadata/color_sequence' in f:
                color_seq = decode_text(f['/metadata/color_sequence'][()])
                color_source = 'hdf5_dataset'
            elif 'color_sequence' in f.get('/metadata', {}).attrs:
                color_seq = decode_text(f['/metadata'].attrs['color_sequence'])
                color_source = 'hdf5_attr'
            elif args.assume_sequence_by_episode_index and ep >= 0 and ep < len(seqs):
                color_seq = seqs[ep]
                color_source = 'sequence_file_index_assumption'

            rows.append({
                'episode': ep,
                'num_steps': num_steps,
                'color_sequence': color_seq,
                'color_source': color_source,
                'pair_at_t0': f"{left[0]}{right[0]}" if num_steps > 0 else 'HH',
                'num_transitions': len(transitions),
                'transitions_json': json.dumps(transitions, ensure_ascii=False),
            })

    with open(args.output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {args.output_csv}")
    print(f"Episodes: {len(rows)}")
    missing_color = sum(1 for r in rows if not r['color_sequence'])
    print(f"Color sequence missing in output: {missing_color}/{len(rows)}")


if __name__ == '__main__':
    main()
