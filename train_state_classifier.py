import argparse
import os
import glob
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from tqdm import tqdm
from PIL import Image

# Constants for States
STATE_HOLD = 0
STATE_INDEP = 1
STATE_COOP = 2

STATE_NAMES = {0: 'HOLD', 1: 'INDEP', 2: 'COOP'}

class DualStateClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super(DualStateClassifier, self).__init__()
        self.backbone = models.resnet18(pretrained=True)
        num_ftrs = self.backbone.fc.in_features
        # Remove original fc
        self.backbone.fc = nn.Identity()
        
        # Dual Heads
        self.fc_left = nn.Linear(num_ftrs, num_classes)
        self.fc_right = nn.Linear(num_ftrs, num_classes)
        
    def forward(self, x):
        features = self.backbone(x)
        out_left = self.fc_left(features)
        out_right = self.fc_right(features)
        return out_left, out_right

class StateDataset(Dataset):
    def __init__(self, dataset_dirs, lookahead=50, transform=None):
        self.samples = [] # (file_path, frame_idx, label_l, label_r)
        self.transform = transform
        self.lookahead = lookahead

        for d in dataset_dirs:
            files = sorted(glob.glob(os.path.join(d, 'episode_*.hdf5')))
            print(f"Scanning {len(files)} episodes in {d}...")
            
            for file_path in tqdm(files, desc=f"Loading Metadata from {os.path.basename(d)}"):
                try:
                    with h5py.File(file_path, 'r') as f:
                        if 'metadata' not in f:
                            continue
                        
                        # Use length of images or qpos
                        if 'observations/images/top' not in f:
                             continue
                        num_frames = f['observations/images/top'].shape[0]
                        
                        if num_frames <= lookahead:
                            continue

                        # Construct frame-wise labels (0 = HOLD default)
                        labels_l = np.zeros(num_frames, dtype=int)
                        labels_r = np.zeros(num_frames, dtype=int)
                        
                        l_segs = f['metadata/left_segments'][()]
                        r_segs = f['metadata/right_segments'][()]
                        
                        def decode(s):
                            return s.decode('utf-8') if isinstance(s, bytes) else s

                        # Fill Left Labels
                        for seg in l_segs:
                            stype = decode(seg['type'])
                            start = max(0, int(seg['start']))
                            end = min(num_frames, int(seg['end']))
                            
                            val = STATE_HOLD
                            if stype == 'independent': val = STATE_INDEP
                            elif stype == 'cooperative': val = STATE_COOP
                            
                            if start < end:
                                labels_l[start:end] = val
                                
                        # Fill Right Labels
                        for seg in r_segs:
                            stype = decode(seg['type'])
                            start = max(0, int(seg['start']))
                            end = min(num_frames, int(seg['end']))
                            
                            val = STATE_HOLD
                            if stype == 'independent': val = STATE_INDEP
                            elif stype == 'cooperative': val = STATE_COOP
                            
                            if start < end:
                                labels_r[start:end] = val
                        
                        # Generate samples with lookahead
                        # Only up to num_frames - lookahead
                        valid_end = num_frames - lookahead
                        
                        # Stride to reduce data redundancy
                        stride = 5 
                        
                        for t in range(0, valid_end, stride):
                            target_t = t + lookahead
                            lbl_l = labels_l[target_t]
                            lbl_r = labels_r[target_t]
                            self.samples.append((file_path, t, lbl_l, lbl_r))
                            
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

        print(f"Total samples collected: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, t, lbl_l, lbl_r = self.samples[idx]

        try:
            with h5py.File(file_path, 'r') as f:
                image_np = f['observations/images/top'][t]
                image = Image.fromarray(image_np.astype('uint8'))
                
                if self.transform:
                    image = self.transform(image)
                
                return image, lbl_l, lbl_r
        except Exception as e:
            print(f"Error loading {file_path} at {t}: {e}")
            return self.__getitem__(np.random.randint(0, len(self.samples)))

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='cuda'):
    best_acc_avg = 0.0
    best_model_wts = model.state_dict()

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader

            running_loss = 0.0
            running_corrects_l = 0
            running_corrects_r = 0

            for inputs, lbl_l, lbl_r in tqdm(dataloader, desc=phase):
                inputs = inputs.to(device)
                lbl_l = lbl_l.to(device)
                lbl_r = lbl_r.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    out_l, out_r = model(inputs)
                    _, preds_l = torch.max(out_l, 1)
                    _, preds_r = torch.max(out_r, 1)
                    
                    loss_l = criterion(out_l, lbl_l)
                    loss_r = criterion(out_r, lbl_r)
                    loss = loss_l + loss_r

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects_l += torch.sum(preds_l == lbl_l.data)
                running_corrects_r += torch.sum(preds_r == lbl_r.data)

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc_l = running_corrects_l.double() / len(dataloader.dataset)
            epoch_acc_r = running_corrects_r.double() / len(dataloader.dataset)
            epoch_acc_avg = (epoch_acc_l + epoch_acc_r) / 2.0

            print(f'{phase} Loss: {epoch_loss:.4f} Acc L: {epoch_acc_l:.4f} Acc R: {epoch_acc_r:.4f} Avg: {epoch_acc_avg:.4f}')

            if phase == 'val' and epoch_acc_avg > best_acc_avg:
                best_acc_avg = epoch_acc_avg
                best_model_wts = model.state_dict()
                torch.save(model.state_dict(), 'state_classifier_best.pth')

        print()

    print(f'Best Val Avg Acc: {best_acc_avg:4f}')
    model.load_state_dict(best_model_wts)
    return model

def main():
    parser = argparse.ArgumentParser(description='Train Dual Arm State Classifier (t+50)')
    parser.add_argument('--dataset_dirs', nargs='+', required=True, help='List of dataset directories')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lookahead', type=int, default=50, help='Steps to look ahead for label')
    
    args = parser.parse_args()

    # Transforms
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    print(f"Loading datasets with lookahead={args.lookahead}...")
    dataset = StateDataset(
        dataset_dirs=args.dataset_dirs,
        lookahead=args.lookahead,
        transform=data_transforms
    )

    if len(dataset) == 0:
        print("No samples found. Check dataset paths.")
        return

    # Split
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = DualStateClassifier(num_classes=3)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=args.epochs, device=device)

if __name__ == '__main__':
    main()
