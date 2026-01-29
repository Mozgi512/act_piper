
import os
import csv
import re
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

def saturation_model(x, a, c):
    # Monotonically increasing function starting at 0: y = a * (1 - exp(-c * x))
    # a: Asymptote (max performance)
    # c: Rate
    return a * (1 - np.exp(-c * x))

def get_epoch_success_rates(ckpt_dir):
    csv_path = os.path.join(ckpt_dir, 'evaluation_results.csv')
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found.")
        return [], []
    
    epochs = []
    success_rates = []
    
    pattern = re.compile(r'policy_epoch_(\d+)_seed')
    
    try:
        with open(csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                ckpt_name = row['Checkpoint']
                match = pattern.search(ckpt_name)
                if match:
                    epoch = int(match.group(1))
                    try:
                        sr = float(row['Success Rate'])
                        epochs.append(epoch)
                        success_rates.append(sr * 100.0) # Convert to Percentage
                    except ValueError:
                        continue
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return [], []
        
    # Sort by epoch
    if epochs:
        # Check if (0,0) exists, if not add it to anchor the data
        if 0 not in epochs:
            epochs.append(0)
            success_rates.append(0.0)
            
        zipped = sorted(zip(epochs, success_rates))
        epochs, success_rates = zip(*zipped)
        return np.array(epochs), np.array(success_rates)
    return np.array([]), np.array([])

def main():
    # Map Number of Episodes -> Checkpoint Directory
    data_sources = {
        50: 'ckpt/cooperation_50eps',
        75: 'ckpt/cooperation_75eps',
        100: 'ckpt/cooperation',
        150: 'ckpt/cooperation_150eps'
    }

    # Publication-ready settings
    plt.rcParams.update({
        'font.size': 14,
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'lines.linewidth': 2.5
    })

    fig, ax = plt.subplots(figsize=(10, 7))

    print("Aggregating results...")
    # Use a high-contrast palette
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'] 
    
    for i, (n_episodes, ckpt_dir) in enumerate(sorted(data_sources.items())):
        epochs, success_rates = get_epoch_success_rates(ckpt_dir)
        if len(epochs) > 0:
            print(f"Plotting for {n_episodes} episodes ({len(epochs)} points)")
            color = colors[i % len(colors)]
            label = f'{n_episodes} Episodes'

            # 1. Plot Raw Data as Scatter Points (Darker)
            ax.scatter(epochs, success_rates, color=color, alpha=0.9, s=25, edgecolors='none')

            # 2. Fit Monotonic Curve (Zero-Intercept Exponential Saturation)
            x_data = epochs
            y_data = success_rates
            
            # Initial guess: a=100, c=1e-4
            p0 = [100, 1e-4] 
            # bounds: a>=0, c>=0
            try:
                popt, pcov = curve_fit(saturation_model, x_data, y_data, p0=p0, bounds=(0, [100, 1]), maxfev=10000)
                
                # Generate smooth points
                xp = np.linspace(0, x_data.max(), 100) # Start from 0
                yp = saturation_model(xp, *popt)
                
                ax.plot(xp, yp, color=color, linestyle='-', label=label)
            except Exception as e:
                print(f"Curve fitting failed for {n_episodes} episodes: {e}")
                # Fallback
                ax.plot(x_data, y_data, color=color, linestyle=':', alpha=0.5, label=f'{label} (raw)')

        else:
            print(f"No data for {n_episodes} episodes")

    # Styling
    ax.set_xlabel('Training Epochs')
    ax.set_ylabel('Success Rate (%)')
    
    # Clean grid
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_ylim(-5, 105)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.legend(loc='lower right', frameon=True, fancybox=False, edgecolor='k')
    
    output_path = 'training_curves_paper.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")

if __name__ == '__main__':
    main()
