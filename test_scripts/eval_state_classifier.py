import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from train_state_classifier import DualStateClassifier, StateDataset, STATE_NAMES

def visualize_predictions(model, dataset, device, num_samples=5):
    model.eval()
    
    # Pick random indices
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    fig, axes = plt.subplots(num_samples, 1, figsize=(8, 4 * num_samples))
    if num_samples == 1: axes = [axes]
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            image_t, lbl_l, lbl_r = dataset[idx]
            
            # Prepare input
            input_t = image_t.unsqueeze(0).to(device)
            
            # Predict
            out_l, out_r = model(input_t)
            _, pred_l = torch.max(out_l, 1)
            _, pred_r = torch.max(out_r, 1)
            
            # Convert image for display (Un-normalize)
            img_np = image_t.permute(1, 2, 0).numpy()
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_np = std * img_np + mean
            img_np = np.clip(img_np, 0, 1)
            
            ax = axes[i]
            ax.imshow(img_np)
            
            gt_l_str = STATE_NAMES[lbl_l.item()]
            gt_r_str = STATE_NAMES[lbl_r.item()]
            pred_l_str = STATE_NAMES[pred_l.item()]
            pred_r_str = STATE_NAMES[pred_r.item()]
            
            # Color code
            color_l = 'green' if pred_l == lbl_l else 'red'
            color_r = 'green' if pred_r == lbl_r else 'red'
            
            title = f"GT: L={gt_l_str}, R={gt_r_str}\n"
            title += f"Pred: L={pred_l_str}, R={pred_r_str}"
            
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.axis('off')
            
            # Add text on image for easier reading?
            # ax.text(10, 30, f"L: {pred_l_str}", color=color_l, fontsize=12, backgroundcolor='white')
            # ax.text(250, 30, f"R: {pred_r_str}", color=color_r, fontsize=12, backgroundcolor='white')

    plt.tight_layout()
    plt.show()

def evaluate_accuracy(model, dataloader, device):
    model.eval()
    correct_l = 0
    correct_r = 0
    total = 0
    
    # Confusion Matrix?
    # minimal setup
    
    with torch.no_grad():
        for inputs, lbl_l, lbl_r in dataloader:
            inputs = inputs.to(device)
            lbl_l = lbl_l.to(device)
            lbl_r = lbl_r.to(device)
            
            out_l, out_r = model(inputs)
            _, pred_l = torch.max(out_l, 1)
            _, pred_r = torch.max(out_r, 1)
            
            correct_l += (pred_l == lbl_l).sum().item()
            correct_r += (pred_r == lbl_r).sum().item()
            total += inputs.size(0)
            
    acc_l = correct_l / total
    acc_r = correct_r / total
    print(f"Evaluation Results ({total} samples):")
    print(f"  Left Accuracy:  {acc_l:.4f}")
    print(f"  Right Accuracy: {acc_r:.4f}")
    print(f"  Average:        {(acc_l+acc_r)/2:.4f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dirs', nargs='+', required=True)
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to .pth file')
    parser.add_argument('--lookahead', type=int, default=50)
    parser.add_argument('--visualize', action='store_true', help='Show plots')
    parser.add_argument('--num_samples', type=int, default=5)
    
    args = parser.parse_args()
    
    # Setup Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load Model
    model = DualStateClassifier(num_classes=3)
    model.load_state_dict(torch.load(args.checkpoint))
    model.to(device)
    model.eval()
    print(f"Model loaded from {args.checkpoint}")
    
    # Dataset
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    dataset = StateDataset(args.dataset_dirs, lookahead=args.lookahead, transform=data_transforms)
    
    # Visualize
    if args.visualize:
        visualize_predictions(model, dataset, device, num_samples=args.num_samples)
    
    # Full Evaluation
    torch.manual_seed(0)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    _, val_dataset = random_split(dataset, [train_size, val_size])
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    evaluate_accuracy(model, val_loader, device)

if __name__ == '__main__':
    main()
