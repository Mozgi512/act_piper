import pandas as pd
import matplotlib.pyplot as plt
import ast
import os

def plot_max_reward_evolution(csv_path="evaluation_results_plot.csv"):
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    
    # Filter out 'best' if present for the main line plot, or handle it separately
    df_numeric = df[df['Epoch'] != 'best'].copy()
    df_numeric['Epoch'] = pd.to_numeric(df_numeric['Epoch'])
    df_numeric = df_numeric.sort_values('Epoch')
    
    # Calculate Average Max Rewards
    all_rewards_data = []
    epochs = []
    
    for _, row in df_numeric.iterrows():
        try:
            # Parse string representation of list "[4.0, 3.0, ...]"
            rewards_list = ast.literal_eval(row['AllMaxRewards'])
            if rewards_list:
                all_rewards_data.append(rewards_list)
                epochs.append(row['Epoch'])
        except Exception as e:
            print(f"Skipping epoch {row['Epoch']}: {e}")
            
    if not epochs:
        print("No valid data found to plot.")
        return

    # IEEE / ICRA styling
    import matplotlib
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    matplotlib.rcParams['font.family'] = 'serif'
    # matplotlib.rcParams['font.serif'] = ['Times New Roman'] # Removed to avoid error on Linux
    matplotlib.rcParams['font.size'] = 12
    matplotlib.rcParams['axes.labelsize'] = 14
    matplotlib.rcParams['legend.fontsize'] = 12
    matplotlib.rcParams['xtick.labelsize'] = 12
    matplotlib.rcParams['ytick.labelsize'] = 12
    
    # Figure size: IEEE column width is ~3.5 inches, double column ~7.
    plt.figure(figsize=(8, 5)) 
    plt.ylim(-1, 7.0) # Adjusted ylim
    
    # Calculate Mean and Std
    means = []
    stds = []
    import numpy as np
    
    for rewards in all_rewards_data:
        means.append(np.mean(rewards))
        stds.append(np.std(rewards))
        
    # Create Error Bar Plot (I-bar)
    plt.errorbar(epochs, means, yerr=stds, fmt='o', 
                 color='blue', # Line color
                 markerfacecolor='black', markeredgecolor='black', markersize=3,
                 ecolor='black', elinewidth=1.5, capsize=4, 
                 linestyle='-', linewidth=1.5,
                 label='Mean ± Std Dev', zorder=10)

    # Process 'best' row if exists
    best_row = df[df['Epoch'] == 'best']
    if not best_row.empty:
        try:
            rewards_list = ast.literal_eval(best_row.iloc[0]['AllMaxRewards'])
            if rewards_list:
                best_mean = np.mean(rewards_list)
                best_std = np.std(rewards_list)
                # Plot Best as a horizontal line or point? Usually line for baseline.
                plt.axhline(y=best_mean, color='red', linestyle='--', linewidth=2, label=f'Best Policy ({best_mean:.2f})')
                # Optional: specific point for best at the end?
                # For now, line is standard.
        except:
            pass

    # plt.title('Max Reward Distribution per Epoch') # No titles in paper figures
    plt.xlabel('Training Epochs')
    plt.ylabel('Processed Objects')
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.legend(frameon=True, fancybox=False, edgecolor='black', loc='lower right')
    plt.tight_layout()
    
    output_png = "processed_objects_errorbar.png"
    plt.savefig(output_png, dpi=300)
    print(f"Plot saved to {output_png}")

if __name__ == "__main__":
    plot_max_reward_evolution()
