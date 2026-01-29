
import os
import argparse
import re
import glob

def cleanup_checkpoints(ckpt_dir):
    if not os.path.isdir(ckpt_dir):
        print(f"Directory not found: {ckpt_dir}")
        return

    print(f"Cleaning up checkpoints in {ckpt_dir}...")
    
    # Matches policy_epoch_100_seed_0.ckpt
    pattern = re.compile(r'policy_epoch_(\d+)_seed_\d+\.ckpt')
    
    files = glob.glob(os.path.join(ckpt_dir, "*.ckpt"))
    deleted_count = 0
    kept_count = 0
    
    for fpath in files:
        fname = os.path.basename(fpath)
        
        # Always keep best and last
        if 'best' in fname or 'last' in fname:
            kept_count += 1
            print(f"Keeping special ckpt: {fname}")
            continue
            
        match = pattern.match(fname)
        if match:
            epoch = int(match.group(1))
            if epoch % 1000 != 0:
                print(f"Deleting {fname} (Epoch {epoch})")
                os.remove(fpath)
                deleted_count += 1
            else:
                print(f"Keeping {fname} (Epoch {epoch})")
                kept_count += 1
        else:
            # Keep unrecognized files (safety)
            print(f"Skipping unrecognized file: {fname}")
            kept_count += 1
            
    print(f"\nCleanup finished.")
    print(f"Deleted: {deleted_count}")
    print(f"Kept: {kept_count}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', type=str, required=True, help='Path to checkpoint directory')
    args = parser.parse_args()
    
    cleanup_checkpoints(args.ckpt_dir)
