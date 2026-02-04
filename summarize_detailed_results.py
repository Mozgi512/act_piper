import pandas as pd
import os

def summarize():
    csv_file = 'detailed_results.csv'
    if not os.path.exists(csv_file):
        print(f"File {csv_file} not found.")
        return

    try:
        df = pd.read_csv(csv_file)
        print(f"Loaded {len(df)} episodes from {csv_file}")
        
        # Calculate mean for Cube columns
        cube_cols = [c for c in df.columns if c.startswith('Cube_')]
        
        # Sort cols for display
        cube_cols.sort(key=lambda x: int(x.split('_')[1]))
        
        summary = df[cube_cols].mean() * 100
        
        print("\n=== Per-Object Success Rates (%) ===")
        print(summary.to_string(float_format="%.1f"))
        
        # Also Total Stats
        print("\n=== Overall Stats ===")
        print(f"Average Total Reward: {df['TotalReward'].mean():.2f}")
        print(f"Average Return: {df['Return'].mean():.2f}")

        # Save summary
        summary.to_csv('summary_per_object.csv', header=['SuccessRate'])
        print("\nSaved summary to summary_per_object.csv")
        
    except Exception as e:
        print(f"Error processing CSV: {e}")

if __name__ == "__main__":
    summarize()
