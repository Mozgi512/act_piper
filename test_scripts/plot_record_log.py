
import matplotlib.pyplot as plt
import csv
import os

def main():
    if not os.path.exists('record_log.csv'):
        print("record_log.csv not found")
        return

    x_success, y_success = [], []
    x_fail, y_fail = [], []

    try:
        with open('record_log.csv', 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if not row: continue
                # format: episode_idx, red_x, red_y, red_z, success
                try:
                    x = float(row[1])
                    y = float(row[2])
                    success = int(float(row[4]))
                    
                    if success == 1:
                        x_success.append(x)
                        y_success.append(y)
                    else:
                        x_fail.append(x)
                        y_fail.append(y)
                except ValueError:
                    continue
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    plt.figure(figsize=(10, 8))
    
    # Plot Success (Green)
    plt.scatter(x_success, y_success, c='green', label='Success', alpha=0.7, s=40, edgecolors='k', linewidth=0.5)
    
    # Plot Failure (Red)
    plt.scatter(x_fail, y_fail, c='red', label='Failure', alpha=0.7, s=40, edgecolors='k', linewidth=0.5)
    
    plt.xlabel('Red Box X Position (m)')
    plt.ylabel('Red Box Y Position (m)')
    plt.title('Task Success vs Initial Red Box Position')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axis('equal') # Preserve spatial aspect ratio
    
    output_path = 'record_log_plot.png'
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")
    print(f"Total: {len(x_success) + len(x_fail)}")
    print(f"Success: {len(x_success)}")
    print(f"Fail: {len(x_fail)}")

if __name__ == '__main__':
    main()
