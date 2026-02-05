import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import h5py
from torchvision import transforms
from train_state_classifier import StateDataset, STATE_NAMES

def inspect_labels(dataset, num_samples=5):
    # Pick random indices
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    fig, axes = plt.subplots(num_samples, 1, figsize=(8, 5 * num_samples))
    if num_samples == 1: axes = [axes]
    
    for i, idx in enumerate(indices):
        file_path, t, lbl_l, lbl_r = dataset.samples[idx]
        
        # Load image directly to print info even if transform might change it
        # (Dataset __getitem__ returns transformed)
        # We'll use dataset[idx] to test the pipeline but we also want metadata info
        
        image_t, _, _ = dataset[idx]
        
        # Un-normalize
        img_np = image_t.permute(1, 2, 0).numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = std * img_np + mean
        img_np = np.clip(img_np, 0, 1)
        
        ax = axes[i]
        ax.imshow(img_np)
        
        l_str = STATE_NAMES[lbl_l]
        r_str = STATE_NAMES[lbl_r]
        
        target_t = t + dataset.lookahead
        
        title = f"File: .../{file_path.split('/')[-1]}\n"
        title += f"Current Frame: {t} -> Label Target Frame: {target_t}\n"
        title += f"GT Left: {l_str} ({lbl_l}) | GT Right: {r_str} ({lbl_r})"
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')
        
        # Print detailed metadata for this file to console
        print(f"\n--- Sample {i+1} ---")
        print(f"File: {file_path}")
        print(f"Current T: {t}, Target T: {target_t}")
        print(f"Labels: L={l_str}, R={r_str}")
        
        # Open file to check segments
        with h5py.File(file_path, 'r') as f:
            total_frames = f['action'].shape[0]
            print(f"Total Frames: {total_frames}")
            
            print("Left Segments:")
            for s in f['metadata/left_segments'][()]:
                stype = s['type'].decode('utf-8')
                print(f"  {stype}: {s['start']} - {s['end']}")
                
            print("Right Segments:")
            for s in f['metadata/right_segments'][()]:
                stype = s['type'].decode('utf-8')
                print(f"  {stype}: {s['start']} - {s['end']}")
        
        # Check if Target T falls into any segment
        print(f"Verification for T={target_t}:")
        
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dirs', nargs='+', required=True)
    parser.add_argument('--lookahead', type=int, default=50)
    parser.add_argument('--num_samples', type=int, default=5)
    
    args = parser.parse_args()
    
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    print(f"Loading dataset with lookahead={args.lookahead}...")
    dataset = StateDataset(args.dataset_dirs, lookahead=args.lookahead, transform=data_transforms)
    
    inspect_labels(dataset, num_samples=args.num_samples)

if __name__ == '__main__':
    main()
