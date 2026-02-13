import os
import re
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models
from tqdm import tqdm
from PIL import Image

class MultiHeadResNet18(nn.Module):
    def __init__(self, num_modes=2, num_objs=7):
        super(MultiHeadResNet18, self).__init__()
        self.backbone = models.resnet18(pretrained=True)
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity() # Remove original fc
        
        # 4 Heads
        self.fc_l_mode = nn.Linear(num_ftrs, num_modes)
        self.fc_r_mode = nn.Linear(num_ftrs, num_modes)
        self.fc_l_obj = nn.Linear(num_ftrs, num_objs)
        self.fc_r_obj = nn.Linear(num_ftrs, num_objs)
        
    def forward(self, x):
        features = self.backbone(x)
        out_lm = self.fc_l_mode(features)
        out_rm = self.fc_r_mode(features)
        out_lo = self.fc_l_obj(features)
        out_ro = self.fc_r_obj(features)
        return out_lm, out_rm, out_lo, out_ro

class MultiHeadTransitionDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = [] # (img_path, l_mode, r_mode, l_obj, r_obj)
        self.transform = transform
        
        # Pattern: LM{m0}_RM{m1}_LO{o0}_RO{o1}
        pattern = re.compile(r'LM(\d)_RM(\d)_LO(\d)_RO(\d)')
        
        for ep_dir in os.listdir(root_dir):
            ep_path = os.path.join(root_dir, ep_dir)
            if not os.path.isdir(ep_path) or not ep_dir.startswith('episode_'):
                continue
            for cls_name in os.listdir(ep_path):
                cls_path = os.path.join(ep_path, cls_name)
                if not os.path.isdir(cls_path): continue
                
                match = pattern.match(cls_name)
                if not match: continue
                
                l_m, r_m, l_o, r_o = map(int, match.groups())
                
                for img_name in os.listdir(cls_path):
                    if img_name.startswith('start_frame_') and img_name.endswith('.png'):
                        self.samples.append((os.path.join(cls_path, img_name), l_m, r_m, l_o, r_o))
        
        print(f"Dataset loaded with {len(self.samples)} samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, lm, rm, lo, ro = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, lm, rm, lo, ro

def train_model(data_dir, output_dir, epochs=20, batch_size=32, lr=1e-4):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    full_dataset = MultiHeadTransitionDataset(data_dir, transform=transform)
    if len(full_dataset) == 0:
        print("Error: No samples found.")
        return

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    model = MultiHeadResNet18().cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')

    for epoch in range(epochs):
        print(f'Epoch {epoch+1}/{epochs}')
        model.train()
        train_loss = 0.0
        
        for inputs, lm, rm, lo, ro in tqdm(train_loader, desc="Training"):
            inputs, lm, rm, lo, ro = inputs.cuda(), lm.cuda(), rm.cuda(), lo.cuda(), ro.cuda()
            
            optimizer.zero_grad()
            out_lm, out_rm, out_lo, out_ro = model(inputs)
            
            loss = criterion(out_lm, lm) + criterion(out_rm, rm) + \
                   criterion(out_lo, lo) + criterion(out_ro, ro)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            
        train_loss /= train_size
        
        model.eval()
        val_loss = 0.0
        corrects = {"lm": 0, "rm": 0, "lo": 0, "ro": 0}
        
        with torch.no_grad():
            for inputs, lm, rm, lo, ro in tqdm(val_loader, desc="Validation"):
                inputs, lm, rm, lo, ro = inputs.cuda(), lm.cuda(), rm.cuda(), lo.cuda(), ro.cuda()
                out_lm, out_rm, out_lo, out_ro = model(inputs)
                
                val_loss += (criterion(out_lm, lm) + criterion(out_rm, rm) + \
                            criterion(out_lo, lo) + criterion(out_ro, ro)).item() * inputs.size(0)
                
                corrects["lm"] += torch.sum(torch.argmax(out_lm, 1) == lm)
                corrects["rm"] += torch.sum(torch.argmax(out_rm, 1) == rm)
                corrects["lo"] += torch.sum(torch.argmax(out_lo, 1) == lo)
                corrects["ro"] += torch.sum(torch.argmax(out_ro, 1) == ro)
        
        val_loss /= val_size
        print(f"Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f}")
        print(f"Accuracies -> LM: {corrects['lm']/val_size:.4f}, RM: {corrects['rm']/val_size:.4f}, LO: {corrects['lo']/val_size:.4f}, RO: {corrects['ro']/val_size:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_multihead_classifier.pth'))
            print("Model saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Path to transition_images')
    parser.add_argument('--output_dir', type=str, default='multihead_classifier_ckpt', help='Output directory')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    
    args = parser.parse_args()
    
    train_model(args.data_dir, args.output_dir, args.epochs, args.batch_size, args.lr)
