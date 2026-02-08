import h5py
import glob
import os
import numpy as np

dataset_dir = '/home/act/act_piper/data/ICTICT'

files = glob.glob(os.path.join(dataset_dir, 'episode_*.hdf5'))
print(f"Found {len(files)} files in {dataset_dir}")

independent_count = 0
cooperative_count = 0
mixed_count = 0
total_frames = 0
independent_frames = 0
cooperative_frames = 0

for file_path in files[:10]: # Check first 10 for detailed view
    try:
        with h5py.File(file_path, 'r') as f:
            if 'metadata' not in f:
                print(f"SKIPPING {file_path}: No metadata")
                continue
            
            l_segs = f['metadata/left_segments'][()]
            r_segs = f['metadata/right_segments'][()]
            num_frames = f['action'].shape[0]
            total_frames += num_frames
            
            # Create a frame-wise mask
            mode_per_frame = np.zeros(num_frames, dtype=int) # 0: Indep, 1: Mixed/Coop
            
            print(f"\n--- {os.path.basename(file_path)} ({num_frames} frames) ---")
            
            # Helper to decode
            def decode(s):
                return s.decode('utf-8') if isinstance(s, bytes) else s

            for seg in l_segs:
                start, end = int(seg['start']), int(seg['end'])
                stype = decode(seg['type'])
                print(f"  L: {start}-{end} : {stype}")
                if stype == 'cooperative':
                    mode_per_frame[start:end] = 1

            for seg in r_segs:
                start, end = int(seg['start']), int(seg['end'])
                stype = decode(seg['type'])
                print(f"  R: {start}-{end} : {stype}")
                if stype == 'cooperative':
                    mode_per_frame[start:end] = 1 # OR logic: if any arm is coop, global is coop?
            
            coop_frames = np.sum(mode_per_frame)
            indep_frames = num_frames - coop_frames
            print(f"  => Indep Frames: {indep_frames}, Coop Frames: {coop_frames}")
            
            
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
