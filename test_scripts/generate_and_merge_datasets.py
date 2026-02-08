
import os
import shutil
import subprocess
import glob

def run_command(command):
    print(f"Running: {command}")
    subprocess.check_call(command, shell=True)

def generate_and_merge():
    base_dir = '/home/act/act_piper/piper_scripted_dataset'
    
    # Config
    phase1_task = 'sim_independent_scripted'
    phase1_dir_base = os.path.join(base_dir, 'sim_independent_2phases')
    phase1_episodes = 100
    
    phase2_task = 'sim_independent_phase2_scripted'
    phase2_dir_base = os.path.join(base_dir, 'sim_independent_phase2_temp') # Temp dir
    phase2_episodes = 100
    
    # Suffixes created by independent_record_sim_episodes with --arm both
    suffixes = ['_left', '_right']
    
    # 1. Clean up temp dirs if exists
    for suffix in suffixes:
        p2_dir = phase2_dir_base + suffix
        if os.path.exists(p2_dir):
            shutil.rmtree(p2_dir)
    
    # 2. Generate Phase 1
    print("--- Generating Phase 1 Data (Both Arms) ---")
    # Note: --dataset_dir will be suffixed by _left and _right inside the script when --arm both
    cmd1 = f"python3 independent_record_sim_episodes.py --task_name {phase1_task} --dataset_dir {phase1_dir_base} --num_episodes {phase1_episodes} --arm both"
    run_command(cmd1)
    
    # 3. Generate Phase 2 (to temp dir)
    print("--- Generating Phase 2 Data (Both Arms) ---")
    cmd2 = f"python3 independent_record_sim_episodes.py --task_name {phase2_task} --dataset_dir {phase2_dir_base} --num_episodes {phase2_episodes} --arm both"
    run_command(cmd2)
    
    # 4. Merge Phase 2 into Phase 1 directory for each arm
    print("--- Merging Phase 2 into Phase 1 Directories ---")
    
    # Calculate start index based on one of the dirs (assuming symmetry)
    phase1_left = phase1_dir_base + '_left'
    existing_files = glob.glob(os.path.join(phase1_left, 'episode_*.hdf5'))
    max_idx = -1
    for f in existing_files:
        basename = os.path.basename(f)
        try:
            idx = int(basename.split('_')[1].split('.')[0])
            if idx > max_idx:
                max_idx = idx
        except:
            pass
            
    start_idx = max_idx + 1
    print(f"Phase 1 has episodes up to {max_idx}. Phase 2 will start from {start_idx}.")
    
    total_count = 0
    for suffix in suffixes:
        p1_dir = phase1_dir_base + suffix
        p2_dir = phase2_dir_base + suffix
        
        print(f"Merging {p2_dir} -> {p1_dir}")
        
        phase2_files = glob.glob(os.path.join(p2_dir, 'episode_*.hdf5'))
        phase2_files.sort()
        
        count = 0
        for f in phase2_files:
             # Calculate new index
             new_idx = start_idx + count
             new_filename = f"episode_{new_idx}.hdf5"
             dst_path = os.path.join(p1_dir, new_filename)
             
             shutil.move(f, dst_path)
             count += 1
        
        total_count = count # efficient enough
        
        # Cleanup temp dir
        if os.path.exists(p2_dir):
            shutil.rmtree(p2_dir)

    print(f"\nSuccess! Combined datasets are in: {phase1_dir_base}_left and _right")
    print(f"Total merged episodes per arm: {start_idx + total_count}")

if __name__ == "__main__":
    # Ensure active environment
    # Note: subprocess inherits env, so just running python3 should work if activated.
    generate_and_merge()
