
import os
import re
import csv

ckpt_dir = 'ckpt/cooperation_50eps_2'
output_csv = os.path.join(ckpt_dir, 'evaluation_results.csv')

# Regular expression to match result filenames and extract epoch
filename_pattern = re.compile(r'result_(policy_epoch_(\d+)_seed_\d+|policy_best)\.txt')

results = []

print(f"Scanning {ckpt_dir} for result files...")

for filename in os.listdir(ckpt_dir):
    match = filename_pattern.match(filename)
    if match:
        ckpt_name = match.group(1) + ".ckpt"
        is_best = 'best' in filename
        epoch = -1
        if not is_best:
            epoch = int(match.group(2))
        else:
             # Handle 'policy_best' usually it doesn't have an epoch in filename, 
             # but we might want it at the end or handle it separately.
             # For sorting, let's treat it as a special case or ignore if we want strictly epoch series.
             # The existing logic usually appends it at the end.
             epoch = 999999999 # Place at the end

        filepath = os.path.join(ckpt_dir, filename)
        with open(filepath, 'r') as f:
            content = f.read()
            
        # Parse content
        # Format:
        # Success rate: 0.0
        # Average return: 0.1
        
        success_rate = None
        avg_return = None
        
        for line in content.split('\n'):
            if 'Success rate:' in line:
                success_rate = float(line.split(':')[1].strip())
            if 'Average return:' in line:
                avg_return = float(line.split(':')[1].strip())
        
        if success_rate is not None and avg_return is not None:
            results.append({
                'epoch': epoch,
                'Checkpoint': ckpt_name,
                'Success Rate': success_rate,
                'Average Return': avg_return
            })

# Sort by epoch
results.sort(key=lambda x: x['epoch'])

# Write to CSV
with open(output_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Checkpoint', 'Success Rate', 'Average Return'])
    
    for res in results:
        writer.writerow([res['Checkpoint'], res['Success Rate'], res['Average Return']])

print(f"Successfully wrote {len(results)} records to {output_csv}")
