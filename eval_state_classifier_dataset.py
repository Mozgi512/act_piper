import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import transforms

from train_state_classifier import DualStateClassifier, STATE_NAMES, StateDataset


def main():
    parser = argparse.ArgumentParser(description="Evaluate dual state classifier on HDF5 dataset")
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--lookahead", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--out_dir", type=str, default="results/state_classifier_eval")
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    dataset = StateDataset(dataset_dirs=[args.dataset_dir], lookahead=args.lookahead, transform=transform)
    if len(dataset) == 0:
        raise RuntimeError("No samples found in dataset")

    print(f"Samples: {len(dataset)}")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )

    model = DualStateClassifier(num_classes=3)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    all_true_l, all_pred_l = [], []
    all_true_r, all_pred_r = [], []

    with torch.no_grad():
        for imgs, lbl_l, lbl_r in loader:
            imgs = imgs.to(device, non_blocking=True)
            lbl_l = lbl_l.to(device, non_blocking=True)
            lbl_r = lbl_r.to(device, non_blocking=True)

            out_l, out_r = model(imgs)
            pred_l = torch.argmax(out_l, dim=1)
            pred_r = torch.argmax(out_r, dim=1)

            all_true_l.extend(lbl_l.cpu().numpy().tolist())
            all_pred_l.extend(pred_l.cpu().numpy().tolist())
            all_true_r.extend(lbl_r.cpu().numpy().tolist())
            all_pred_r.extend(pred_r.cpu().numpy().tolist())

    all_true_l = np.array(all_true_l)
    all_pred_l = np.array(all_pred_l)
    all_true_r = np.array(all_true_r)
    all_pred_r = np.array(all_pred_r)

    acc_l = float((all_true_l == all_pred_l).mean())
    acc_r = float((all_true_r == all_pred_r).mean())
    acc_avg = float((acc_l + acc_r) / 2.0)

    labels = [0, 1, 2]
    class_names = [STATE_NAMES[i] for i in labels]

    cm_l = confusion_matrix(all_true_l, all_pred_l, labels=labels)
    cm_r = confusion_matrix(all_true_r, all_pred_r, labels=labels)

    report_l = classification_report(all_true_l, all_pred_l, labels=labels, target_names=class_names, digits=4)
    report_r = classification_report(all_true_r, all_pred_r, labels=labels, target_names=class_names, digits=4)

    print("\n=== Accuracy ===")
    print(f"Left  : {acc_l:.4f}")
    print(f"Right : {acc_r:.4f}")
    print(f"Avg   : {acc_avg:.4f}")

    print("\n=== Confusion Matrix (Left) ===")
    print(cm_l)
    print("\n=== Confusion Matrix (Right) ===")
    print(cm_r)

    print("\n=== Classification Report (Left) ===")
    print(report_l)
    print("\n=== Classification Report (Right) ===")
    print(report_r)

    np.savetxt(os.path.join(args.out_dir, "confusion_left.csv"), cm_l.astype(int), fmt="%d", delimiter=",")
    np.savetxt(os.path.join(args.out_dir, "confusion_right.csv"), cm_r.astype(int), fmt="%d", delimiter=",")

    summary = {
        "checkpoint": args.ckpt,
        "dataset_dir": args.dataset_dir,
        "lookahead": int(args.lookahead),
        "samples": int(len(dataset)),
        "accuracy_left": acc_l,
        "accuracy_right": acc_r,
        "accuracy_avg": acc_avg,
        "class_order": class_names,
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    for side, cm in [("left", cm_l), ("right", cm_r)]:
        fig, ax = plt.subplots(figsize=(5, 4.5))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(f"Confusion Matrix ({side})")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, f"confusion_{side}.png"), dpi=180)
        plt.close(fig)

    print(f"\nSaved outputs to: {args.out_dir}")


if __name__ == "__main__":
    main()
