import os
import re
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from tqdm import tqdm
import numpy as np
from PIL import Image

class MultiHeadResNet18(nn.Module):
    def __init__(self, num_modes=2, num_objs=7):
        super(MultiHeadResNet18, self).__init__()
        self.backbone = models.resnet18(pretrained=False)
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        self.fc_l_mode = nn.Linear(num_ftrs, num_modes)
        self.fc_r_mode = nn.Linear(num_ftrs, num_modes)
        self.fc_l_obj = nn.Linear(num_ftrs, num_objs)
        self.fc_r_obj = nn.Linear(num_ftrs, num_objs)
        
    def forward(self, x):
        features = self.backbone(x)
        return self.fc_l_mode(features), self.fc_r_mode(features), \
               self.fc_l_obj(features), self.fc_r_obj(features)

class MultiHeadTransitionDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform
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
        return img, lm, rm, lo, ro, path

def evaluate_multihead(data_dir, model_path, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = MultiHeadTransitionDataset(data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    model = MultiHeadResNet18().cuda()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    corrects = {"lm": 0, "rm": 0, "lo": 0, "ro": 0}
    failures = []
    total = 0
    
    MODES = ["INDEP", "COOP/HOLD"]
    OBJS = ["L3", "L2", "L1", "L0", "R0", "R1", "NONE"]

    print(f"Evaluating model: {model_path}")
    
    with torch.no_grad():
        for i, (inputs, lm, rm, lo, ro, paths) in enumerate(tqdm(loader, desc="Evaluating")):
            inputs = inputs.cuda()
            lm, rm, lo, ro = lm.cuda(), rm.cuda(), lo.cuda(), ro.cuda()
            
            out_lm, out_rm, out_lo, out_ro = model(inputs)
            
            p_lm = torch.argmax(out_lm, 1)
            p_rm = torch.argmax(out_rm, 1)
            p_lo = torch.argmax(out_lo, 1)
            p_ro = torch.argmax(out_ro, 1)
            
            corrects["lm"] += torch.sum(p_lm == lm).item()
            corrects["rm"] += torch.sum(p_rm == rm).item()
            corrects["lo"] += torch.sum(p_lo == lo).item()
            corrects["ro"] += torch.sum(p_ro == ro).item()
            total += inputs.size(0)
            
            # Track Failures
            is_correct = (p_lm == lm) & (p_rm == rm) & (p_lo == lo) & (p_ro == ro)
            for j in range(inputs.size(0)):
                if not is_correct[j]:
                    failures.append({
                        "path": paths[j],
                        "true": (lm[j].item(), rm[j].item(), lo[j].item(), ro[j].item()),
                        "pred": (p_lm[j].item(), p_rm[j].item(), p_lo[j].item(), p_ro[j].item())
                    })

    print("\n" + "="*30)
    print(f"FAILED SAMPLES ({len(failures)})")
    for f in failures:
        t_lm, t_rm, t_lo, t_ro = f["true"]
        p_lm, p_rm, p_lo, p_ro = f["pred"]
        print(f"\nImg: {f['path']}")
        print(f"  GT   -> L_Mode:{MODES[t_lm]:10} R_Mode:{MODES[t_rm]:10} L_Obj:{OBJS[t_lo]:5} R_Obj:{OBJS[t_ro]:5}")
        print(f"  PRED -> L_Mode:{MODES[p_lm]:10} R_Mode:{MODES[p_rm]:10} L_Obj:{OBJS[p_lo]:5} R_Obj:{OBJS[p_ro]:5}")
        
        # Point out what exactly failed
        diffs = []
        if t_lm != p_lm: diffs.append("L_Mode")
        if t_rm != p_rm: diffs.append("R_Mode")
        if t_lo != p_lo: diffs.append("L_Obj")
        if t_ro != p_ro: diffs.append("R_Obj")
        print(f"  Diffs: {', '.join(diffs)}")

    print("\n" + "="*30)
    print(f"Accuracy Summary (Total samples: {total})")
    print(f"  Left Arm Mode:  {corrects['lm']/total:.4f}")
    print(f"  Right Arm Mode: {corrects['rm']/total:.4f}")
    print(f"  Left Object ID: {corrects['lo']/total:.4f}")
    print(f"  Right Object ID:{corrects['ro']/total:.4f}")
    print("="*30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()
    evaluate_multihead(args.data_dir, args.model_path, args.batch_size)
