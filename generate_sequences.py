import csv
import itertools

import csv
import itertools

def generate_commands(seq):
    # Parse 10-char sequence to commands
    # Logic:
    # 1. 'xy' pair (g/b) -> 'C'. Sets context (Top/Base).
    #    - gb: Top=R, Base=L
    #    - bg: Top=L, Base=R
    # 2. 'r' after 'xy' -> Alternates Top/Base.
    # 3. 'rr' at start (no context) -> 'I'
    # 4. 'r' at start (no context, odd) -> 'R'
    
    commands = []
    consumed = [False] * len(seq)
    cmd_list = []
    
    # Pass 1: Identify all 'gb'/'bg' pairs (C commands)
    c_ranges = []
    for i in range(len(seq)):
        if consumed[i]:
            continue
            
        if seq[i] in ['g', 'b']:
            # Start of possible pair
            # Find matching pair
            pair_idx = -1
            target = 'b' if seq[i] == 'g' else 'g'
            
            for j in range(i + 1, len(seq)):
                if not consumed[j] and seq[j] == target:
                    pair_idx = j
                    break
            
            if pair_idx != -1:
                # Found pair
                cmd_list.append((i, 'C'))
                c_ranges.append((i, pair_idx))
                consumed[i] = True
                consumed[pair_idx] = True
                
    # Pass 2: Collect all unconsumed 'r' indices
    r_indices = []
    for i in range(len(seq)):
        if not consumed[i] and seq[i] == 'r':
            r_indices.append(i)
            consumed[i] = True
            
    # Pass 3: Greedy Pairing of 'r's (I commands)
    # Pair them up: (r1, r2) -> 'I' at r1
    # Orphan r -> 'R' at r
    
    # Pass 3: Proximity Greedy Pairing with Conditional Boost
    # Pair (r1, r2) -> 'I'
    # Condition 1: Pair if distance is small (<= 2).
    # Condition 2: Priority Boost (-1.5) is CONDITIONAL.
    #   - Apply Boost ONLY if:
    #     1. Head Region (r1 < 5)
    #     2. AND Fully Contained in a C command (c_start < r1 and r2 < c_end).
    #   - Otherwise, use default priority (r1).
    
    HEAD_THRESHOLD = 5
    
    k = 0
    while k < len(r_indices):
        current_r_idx = r_indices[k]
        
        if k + 1 < len(r_indices):
            # Potential pair exists
            next_r_idx = r_indices[k+1]
            
            # Check Proximity
            distance = next_r_idx - current_r_idx
            
            if distance <= 2:
                # Close enough to pair
                
                # Determine Priority
                priority = current_r_idx # Default
                
                # Check for Boost Conditions
                if current_r_idx < HEAD_THRESHOLD:
                    # Check if fully contained in any C range
                    is_contained = False
                    for c_start, c_end in c_ranges:
                        if c_start < current_r_idx and next_r_idx < c_end:
                            is_contained = True
                            break
                    
                    if is_contained:
                        priority = current_r_idx - 1.5
                
                cmd_list.append((priority, 'I'))
                k += 2
            else:
                # Too far apart -> Reject Pairing
                # Treat current r as Single 'R'
                cmd_list.append((current_r_idx, 'R'))
                k += 1
        else:
            # Oracle 'R' (Single)
            cmd_list.append((current_r_idx, 'R'))
            k += 1
            
    # Sort by index
    cmd_list.sort(key=lambda x: x[0])
    return "".join([c[1] for c in cmd_list])

def generate_sequences():
    # Base patterns and their possible expansions
    # (xy) -> gb, bg
    # (xry) -> grb, brg
    # (xrry) -> grrb, brrg
    # (r) -> r
    # (rr) -> rr
    # (rrr) -> rrr
    
    sub_sequences = [
        'gb', 'bg',          # from (xy)
        'grb', 'brg',        # from (xry)
        'grrb', 'brrg',      # from (xrry)
        'r',                 # from (r)
        'rr',                # from (rr)
        'rrr'                # from (rrr)
    ]
    
    valid_sequences = set()
    
    # Target length is 10
    target_length = 10
    
    # DFS to find all combinations
    def backtrack(current_seq):
        if len(current_seq) == target_length:
            valid_sequences.add(current_seq)
            return
        
        if len(current_seq) > target_length:
            return
            
        for sub in sub_sequences:
            backtrack(current_seq + sub)
            
    backtrack("")
    
    # Sort for consistency
    sorted_sequences = sorted(list(valid_sequences))
    
    print(f"Generated {len(sorted_sequences)} unique sequences.")
    
    # Save to CSV
    output_file = 'sequences.csv'
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        # writer.writerow(['sequence', 'commands']) 
        for seq in sorted_sequences:
            cmds = generate_commands(seq)
            writer.writerow([seq, cmds])
            
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    generate_sequences()
