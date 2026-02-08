
import csv
import matplotlib.pyplot as plt
import re
import os
import argparse

def parse_epoch(ckpt_name):
    if 'best' in ckpt_name:
        return -1 # Mark best separately
    match = re.search(r'epoch_(\d+)_', ckpt_name)
    if match:
        return int(match.group(1))
    return None

def plot_results(csv_path, output_path):
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    epochs = []
    success_rates = []
    avg_returns = []

    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        try:
            header = next(reader) # Skip header
        except StopIteration:
            print("Error: CSV file is empty.")
            return
        
        raw_data = []
        for row in reader:
            if not row: continue
            if len(row) < 3: continue 
            ckpt = row[0]
            try:
                sr = float(row[1])
                ret = float(row[2])
                epoch = parse_epoch(ckpt)
                if epoch is not None and epoch != -1:
                    raw_data.append((epoch, sr, ret))
            except ValueError:
                continue
                
    # Sort by epoch
    raw_data.sort(key=lambda x: x[0])
    
    # Unpack
    if not raw_data:
        print("No valid data found.")
        return

    epochs = [x[0] for x in raw_data]
    success_rates = [x[1] for x in raw_data]
    avg_returns = [x[2] for x in raw_data]

    # Plot
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Success Rate', color=color)
    ax1.plot(epochs, success_rates, color=color, marker='o', label='Success Rate')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True)

    # Instantiate a second axes that shares the same x-axis
    ax2 = ax1.twinx()  
    color = 'tab:orange'
    ax2.set_ylabel('Average Return', color=color)
    ax2.plot(epochs, avg_returns, color=color, linestyle='--', marker='x', label='Avg Return')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Evaluation Results: Success Rate & Return Trend')
    fig.tight_layout()
    
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot evaluation results from a CSV file.")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the evaluation_results.csv file")
    parser.add_argument("--output_path", type=str, default=None, help="Path to save the output plot (optional, defaults to same dir as csv)")
    
    args = parser.parse_args()
    
    if args.output_path is None:
        args.output_path = os.path.join(os.path.dirname(args.csv_path), "evaluation_plot.png")
        
    plot_results(args.csv_path, args.output_path)
