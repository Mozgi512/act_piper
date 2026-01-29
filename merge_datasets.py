import os
import shutil
import re

SOURCE_DIR = 'piper_scripted_dataset/cooperation_200'
DEST_DIR = 'piper_scripted_dataset/cooperation'
OFFSET = 100

def main():
    if not os.path.exists(SOURCE_DIR):
        print(f"Source directory {SOURCE_DIR} does not exist.")
        return

    if not os.path.exists(DEST_DIR):
        print(f"Destination directory {DEST_DIR} does not exist.")
        return

    files = os.listdir(SOURCE_DIR)
    print(f"Found {len(files)} files in {SOURCE_DIR}")

    count = 0
    for filename in files:
        # Match episode_N.hdf5, episode_N_qpos.png, episode_N_video.mp4
        match = re.match(r'episode_(\d+)(.+)', filename)
        if match:
            idx = int(match.group(1))
            suffix = match.group(2)
            
            new_idx = idx + OFFSET
            new_filename = f"episode_{new_idx}{suffix}"
            
            src_path = os.path.join(SOURCE_DIR, filename)
            dest_path = os.path.join(DEST_DIR, new_filename)
            
            print(f"Moving {filename} -> {new_filename}")
            shutil.move(src_path, dest_path)
            count += 1
            
    print(f"Moved {count} files.")
    
    # Check if Source is empty
    remaining = os.listdir(SOURCE_DIR)
    if not remaining:
        print(f"Source directory {SOURCE_DIR} is empty. Removing...")
        os.rmdir(SOURCE_DIR)
    else:
        print(f"Source directory {SOURCE_DIR} still contains {len(remaining)} files.")

if __name__ == '__main__':
    main()
