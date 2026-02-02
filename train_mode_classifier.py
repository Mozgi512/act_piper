
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
    def __init__(self, independent_dirs, cooperative_dirs, transform=None):
        self.files = []
        self.labels = []
        self.transform = transform

        # Label 0: Independent
        for d in independent_dirs:
            files = glob.glob(os.path.join(d, 'episode_*.hdf5'))
            self.files.extend(files)
            self.labels.extend([0] * len(files))
            print(f"Found {len(files)} independent episodes in {d}")

        # Label 1: Cooperative
        for d in cooperative_dirs:
            files = glob.glob(os.path.join(d, 'episode_*.hdf5'))
            self.files.extend(files)
            self.labels.extend([1] * len(files))
            print(f"Found {len(files)} cooperative episodes in {d}")

        print(f"Total episodes: {len(self.files)}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        label = self.labels[idx]

        try:
            with h5py.File(file_path, 'r') as f:
                # Get random frame index
                # Assuming 'observations/images/top' exists and has shape (T, H, W, 3)
                if 'observations/images/top' not in f:
                     raise ValueError(f"No top image in {file_path}")
                
                images = f['observations/images/top']
                num_frames = images.shape[0]
                
                # Pick a random frame
                rand_t = np.random.randint(0, num_frames)
                image_np = images[rand_t] # (H, W, 3) or (H, W, 3) depending on storage
                
                # Convert to PIL for transforms
                image = Image.fromarray(image_np.astype('uint8'))
                
                if self.transform:
                    image = self.transform(image)
                
                return image, label
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            # Return a dummy or handle error (here we just retry random another one for simplicity or crash)
            return self.__getitem__(np.random.randint(0, len(self.files)))

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
    parser.add_argument('--independent_dirs', nargs='+', required=True, help='Directories for Independent class')
    parser.add_argument('--cooperative_dirs', nargs='+', required=True, help='Directories for Cooperative class')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--future_steps', type=int, default=0, help='Future steps to predict (currently unused for static labeling)')
    
    args = parser.parse_args()

    # Transforms
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)), # ResNet standard
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Dataset using list of directories
    dataset = ModeClassificationDataset(
        independent_dirs=args.independent_dirs,
        cooperative_dirs=args.cooperative_dirs,
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
