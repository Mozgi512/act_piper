import os
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from tqdm import tqdm

def train_model(data_dir, output_dir, epochs=10, batch_size=32, lr=1e-4):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Data Transformations
    # Standard ResNet preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load Dataset from folders
    # The folders are expected to be organized as: data_dir/class_name/image.png
    # But our extract_transitions.py saves as: data_dir/episode_X/class_name/image.png
    # We need to flatten this or use a custom loader. 
    # Let's write a simple wrapper to collect all images from episode subfolders.
    
    all_images_dataset = []
    # If the user points to a directory containing "episode_*" folders:
    # We crawl one level deeper to find the class folders.
    
    class_to_idx = {}
    class_names = []
    
    # First pass: find all unique class names
    for root, dirs, files in os.walk(data_dir):
        if 'episode_' in root:
            for d in dirs:
                if d not in class_names:
                    class_names.append(d)
    
    class_names.sort()
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    
    # Save class mapping
    with open(os.path.join(output_dir, 'class_map.json'), 'w') as f:
        json.dump(class_to_idx, f, indent=4)
    print(f"Saved class mapping for {len(class_names)} classes: {class_names}")

    # Custom Dataset that crawls episode folders
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
                    idx = class_to_idx[cls_name]
                    for img_name in os.listdir(cls_path):
                        if img_name.startswith('start_frame_') and img_name.endswith('.png'):
                            self.samples.append((os.path.join(cls_path, img_name), idx))
        
        def __len__(self):
            return len(self.samples)
        
        def __getitem__(self, idx):
            path, target = self.samples[idx]
            from PIL import Image
            img = Image.open(path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, target

    full_dataset = TransitionDataset(data_dir, class_to_idx, transform=transform)
    if len(full_dataset) == 0:
        print("Error: No images found. Check your data_dir structure.")
        return

    # Split into Train and Val
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model: ResNet18
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
    model = model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_acc = 0.0

    for epoch in range(epochs):
        print(f'Epoch {epoch+1}/{epochs}')
        
        # Training Phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in tqdm(train_loader, desc="Training"):
            inputs = inputs.cuda()
            labels = labels.cuda()
            
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
        train_loss = running_loss / train_size
        train_acc = running_corrects.double() / train_size
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validation"):
                inputs = inputs.cuda()
                labels = labels.cuda()
                
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)
        
        val_loss = val_loss / val_size
        val_acc = val_corrects.double() / val_size
        
        print(f'Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_classifier.pth'))
            print(f"New best model saved with Acc: {best_acc:.4f}")

    print(f'Training complete. Best Val Acc: {best_acc:4f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing episode folders with classified images')
    parser.add_argument('--output_dir', type=str, default='classifier_ckpt', help='Output directory for model and mapping')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()
    
    train_model(args.data_dir, args.output_dir, args.epochs, args.batch_size, args.lr)
