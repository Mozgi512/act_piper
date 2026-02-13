print("DEBUG: Script started")
import os
import json
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from PIL import Image

def evaluate_model(data_dir, model_path, class_map_path, batch_size=32):
    print("Starting evaluation script...")
    print(f"CUDA available: {torch.cuda.is_available()}")
    # Load Class Mapping
    with open(class_map_path, 'r') as f:
        class_to_idx = json.load(f)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    # Data Transformation (must match training)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Custom Dataset (same as training)
    class TransitionDataset(torch.utils.data.Dataset):
        def __init__(self, root_dir, class_to_idx, transform=None):
            self.samples = []
            self.transform = transform
            for ep_dir in os.listdir(root_dir):
                ep_path = os.path.join(root_dir, ep_dir)
                if not os.path.isdir(ep_path) or not ep_dir.startswith('episode_'):
                    continue
                for cls_name in os.listdir(ep_path):
                    cls_path = os.path.join(ep_path, cls_name)
                    if not os.path.isdir(cls_path): continue
                    if cls_name not in class_to_idx:
                        print(f"Warning: Class {cls_name} not found in class map. Skipping.")
                        continue
                    idx = class_to_idx[cls_name]
                    for img_name in os.listdir(cls_path):
                        if img_name.startswith('start_frame_') and img_name.endswith('.png'):
                            self.samples.append((os.path.join(cls_path, img_name), idx))
        
        def __len__(self):
            return len(self.samples)
        
        def __getitem__(self, idx):
            path, target = self.samples[idx]
            img = Image.open(path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, target

    dataset = TransitionDataset(data_dir, class_to_idx, transform=transform)
    if len(dataset) == 0:
        print("Error: No test images found.")
        return
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Load Model
    model = models.resnet18()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
    
    model.load_state_dict(torch.load(model_path))
    model = model.cuda()
    model.eval()

    all_preds = []
    all_labels = []

    print(f"Evaluating {len(dataset)} images across {len(class_names)} classes...")
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Inference"):
            inputs = inputs.cuda()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Calculate Metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    acc = np.mean(all_preds == all_labels)
    print(f"\nOverall Accuracy: {acc:.4f}")

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(all_labels, all_preds)
    
    # Print clean confusion matrix
    header = " " * 12 + " ".join([f"{name[:8]:>8}" for name in class_names])
    print(header)
    for i, row in enumerate(cm):
        row_str = f"{class_names[i][:11]:>11} " + " ".join([f"{val:8d}" for val in row])
        print(row_str)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--model_path', type=str, default='classifier_ckpt/best_classifier.pth')
    parser.add_argument('--class_map_path', type=str, default='classifier_ckpt/class_map.json')
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()
    
    evaluate_model(args.data_dir, args.model_path, args.class_map_path, args.batch_size)
