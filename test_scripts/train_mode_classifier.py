
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

class ModeClassificationDataset(Dataset):
    def __init__(self, dataset_dirs, transform=None):
        self.samples = [] # List of tuples: (file_path, frame_idx, label)
        self.transform = transform

        for d in dataset_dirs:
            files = glob.glob(os.path.join(d, 'episode_*.hdf5'))
            print(f"Scanning {len(files)} episodes in {d}...")
            
            for file_path in tqdm(files, desc=f"Loading Metadata from {os.path.basename(d)}"):
                try:
                    with h5py.File(file_path, 'r') as f:
                        if 'metadata' not in f:
                            continue
                            
                        # Extract total frames
                        num_frames = f['action'].shape[0] if 'action' in f else 0
                        
                        if num_frames == 0:
                            continue

                        # Construct frame-wise labels
                        # Default is Independent (0)
                        labels = np.zeros(num_frames, dtype=int)
                        
                        # Apply Cooperative (1) logic
                        # "If any arm is cooperative -> 1"
                        
                        l_segs = f['metadata/left_segments'][()]
                        r_segs = f['metadata/right_segments'][()]
                        
                        def decode(s):
                            return s.decode('utf-8') if isinstance(s, bytes) else s

                        for seg in l_segs:
                            stype = decode(seg['type'])
                            if stype == 'cooperative':
                                start, end = int(seg['start']), int(seg['end'])
                                # Clamp to range
                                start = max(0, start)
                                end = min(num_frames, end)
                                labels[start:end] = 1
                        
                        for seg in r_segs:
                            stype = decode(seg['type'])
                            if stype == 'cooperative':
                                start, end = int(seg['start']), int(seg['end'])
                                start = max(0, start)
                                end = min(num_frames, end)
                                labels[start:end] = 1
                        
                        # Store samples
                        # Strategy: Sample ALL frames? Or strided?
                        # For efficiency, let's take a strided sample (e.g. every 10 frames)
                        # or random samples per episode to avoid huge dataset in RAM?
                        # Actually, we store (path, idx, label) metadata list, not images.
                        
                        stride = 5  # Reduce data density slightly
                        for t in range(0, num_frames, stride):
                            self.samples.append((file_path, t, labels[t]))
                            
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

        print(f"Total samples collected: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, t, label = self.samples[idx]

        try:
            with h5py.File(file_path, 'r') as f:
                # Assuming 'observations/images/top' exists
                if 'observations/images/top' not in f:
                     raise ValueError(f"No top image in {file_path}")
                
                # Retrieve specific frame
                image_np = f['observations/images/top'][t]
                
                # Convert to PIL for transforms
                image = Image.fromarray(image_np.astype('uint8'))
                
                if self.transform:
                    image = self.transform(image)
                
                return image, label
        except Exception as e:
            print(f"Error loading {file_path} at {t}: {e}")
            # Retry random
            return self.__getitem__(np.random.randint(0, len(self.samples)))

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='cuda'):
    best_acc = 0.0
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
            running_corrects = 0

            for inputs, labels in tqdm(dataloader, desc=phase):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
                torch.save(model.state_dict(), 'mode_classifier_best.pth')

        print()

    print(f'Best Val Acc: {best_acc:4f}')
    model.load_state_dict(best_model_wts)
    return model

def main():
    parser = argparse.ArgumentParser(description='Train Task Mode Classifier')
    parser.add_argument('--dataset_dirs', nargs='+', required=True, help='List of dataset directories (e.g. data/ICTICT)')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    
    args = parser.parse_args()

    # Transforms
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)), # ResNet standard
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Dataset using list of directories
    dataset = ModeClassificationDataset(
        dataset_dirs=args.dataset_dirs,
        transform=data_transforms
    )

    # Split
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2) # Binary classification
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=args.epochs, device=device)

if __name__ == '__main__':
    main()
