import argparse
import csv
import json
import os
from collections import defaultdict


def load_sequences(sequence_file):
    rows = []
    with open(sequence_file, 'r') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if not row:
                continue
            color = row[0].strip() if len(row) >= 1 else ''
            cmd = row[1].strip().upper() if len(row) >= 2 else ''
            if cmd:
                rows.append({'row': i, 'color': color, 'cmd': cmd})
    return rows


def parse_gt_row(row):
    pair0 = (row.get('pair_at_t0') or '').strip()
    transitions_json = row.get('transitions_json') or '[]'
    try:
        transitions = json.loads(transitions_json)
    except Exception:
        transitions = []

    timeline = []
    if pair0:
        timeline.append(pair0)
    for item in transitions:
        pair = (item.get('pair') or '').strip()
        if pair:
            timeline.append(pair)

    # compress consecutive duplicates
    compressed = []
    for p in timeline:
        if not compressed or compressed[-1] != p:
            compressed.append(p)

    # coarse features from mode timeline
    features = {
        'cc_runs': 0,
        'ii_runs': 0,
        'mixed_runs': 0,
        'hold_runs': 0,
    }

    def classify(pair):
        left = pair[0] if len(pair) >= 1 else 'H'
        right = pair[1] if len(pair) >= 2 else 'H'
        if left == 'C' and right == 'C':
            return 'cc_runs'
        if left == 'I' and right == 'I':
            return 'ii_runs'
        if ('C' in (left, right)) and ('I' in (left, right)):
            return 'mixed_runs'
        return 'hold_runs'

    for p in compressed:
        features[classify(p)] += 1

    features['signature'] = '>'.join(compressed)
    return features


def command_features(cmd):
    cmd = cmd.upper()
    return {
        'c_count': cmd.count('C') + cmd.count('T') + cmd.count('B'),
        'i_count': cmd.count('I'),
        'm_count': cmd.count('R') + cmd.count('L'),
        'len': len(cmd),
    }


def score_match(gtf, cf):
    # heuristic distance: smaller is better
    return (
        abs(gtf['cc_runs'] - cf['c_count']) * 3
        + abs(gtf['ii_runs'] - cf['i_count']) * 2
        + abs(gtf['mixed_runs'] - cf['m_count']) * 3
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_csv', required=True)
    parser.add_argument('--sequence_file', required=True)
    parser.add_argument('--output_csv', default='gt_to_sequence_candidates.csv')
    parser.add_argument('--top_k', type=int, default=5)
    args = parser.parse_args()

    seq_rows = load_sequences(args.sequence_file)
    if not seq_rows:
        raise RuntimeError('No sequence rows with command column found.')

    seq_cmd_features = []
    for s in seq_rows:
        seq_cmd_features.append({**s, **command_features(s['cmd'])})

    outputs = []
    with open(args.gt_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            gtf = parse_gt_row(row)

            scored = []
            for s in seq_cmd_features:
                sc = score_match(gtf, s)
                scored.append((sc, s))
            scored.sort(key=lambda x: x[0])

            top = scored[: max(1, args.top_k)]
            out = {
                'episode': row.get('episode', ''),
                'mode_signature': gtf['signature'],
                'cc_runs': gtf['cc_runs'],
                'ii_runs': gtf['ii_runs'],
                'mixed_runs': gtf['mixed_runs'],
            }
            for idx, (sc, s) in enumerate(top, start=1):
                out[f'cand{idx}_score'] = sc
                out[f'cand{idx}_row'] = s['row']
                out[f'cand{idx}_color'] = s['color']
                out[f'cand{idx}_cmd'] = s['cmd']
            outputs.append(out)

    if not outputs:
        raise RuntimeError('No GT rows found.')

    # normalize fieldnames
    fieldnames = []
    for k in outputs[0].keys():
        fieldnames.append(k)

    with open(args.output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(outputs)

    print(f'Saved: {args.output_csv}')
    print(f'Episodes: {len(outputs)}')


if __name__ == '__main__':
    main()
