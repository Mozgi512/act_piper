
import os
import re

def main():
    ckpt_base_dir = '/home/act/act_piper/ckpt'
    # Pattern to match: policy_epoch_{epoch}_seed_{seed}.ckpt
    pattern = re.compile(r'policy_epoch_(\d+)_seed_\d+\.ckpt')
    
    deleted_count = 0
    
    if not os.path.exists(ckpt_base_dir):
        print(f"Directory {ckpt_base_dir} does not exist.")
        return

    print(f"Scanning {ckpt_base_dir} for checkpoints to cleanup...")

    for root, dirs, files in os.walk(ckpt_base_dir):
        for filename in files:
            match = pattern.match(filename)
            if match:
                epoch = int(match.group(1))
                # Delete if NOT a multiple of 1000
                if epoch % 1000 != 0:
                    filepath = os.path.join(root, filename)
                    try:
                        os.remove(filepath)
                        # print(f"Deleted: {filepath}")
                        deleted_count += 1
                    except Exception as e:
                        print(f"Error deleting {filepath}: {e}")
    
    print(f"Cleanup complete. Deleted {deleted_count} files.")

if __name__ == "__main__":
    main()
