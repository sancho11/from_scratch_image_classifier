"""
evaluate.py

Compute evaluation metrics for a trained model:
  1) Confusion matrix on validation set (saved as confusion_matrix.png)
  2) Submission CSV on test set (saved as submission.csv)

Usage:
    python evaluate.py /path/to/checkpoint.pt \
        --data_root /path/to/dataset \
        --output_dir /path/to/eval_outputs \
        [--batch_size 32]
"""

import os, glob
import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torchvision import datasets
from PIL import Image

from common import (
    CustomCNN,
    TrainingConfig,
    SystemConfig,
    get_data,
    image_common_transforms,
    get_mean_std
)

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained image classifier")
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to the .pt model checkpoint file",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="./dataset",
        help="Root directory of dataset (contains Train/, Valid/, Test/)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Where to save confusion_matrix.png and submission.csv",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for validation and test loaders",
    )
    return parser.parse_args()


def evaluate_confusion_matrix(model, valid_loader, classes, device, out_path):
    model.to(device).eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for imgs, targets in valid_loader:
            imgs = imgs.to(device)
            targets = targets.to(device)
            logits = model(imgs)
            preds = logits.argmax(dim=1)
            y_true.extend(targets.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=classes)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, xticks_rotation=45)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[✓] Confusion matrix saved to {out_path}")


class FlatImageDataset(datasets.ImageFolder):
    """
    ImageFolder subclass that scans a directory (and subdirectories) for files
    matching given extensions, returning (image, filename) pairs.

    Args:
        root (str): Root directory containing images in any nested structure.
        transform (callable, optional): Transform to apply to PIL images.
        exts (tuple[str], optional): File extensions to include.
    """
    def __init__(
        self,
        root: str,
        transform=None,
        exts: tuple = (".jpg", ".jpeg", ".png", ".bmp", ".gif"),
    ):
        # Collect all file paths recursively matching extensions
        self.files = []
        for ext in exts:
            self.files.extend(glob.glob(os.path.join(root, f"**/*{ext}"), recursive=True))
        self.transform = transform

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        path = self.files[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        filename = os.path.basename(path)
        return img, filename


def generate_submission(model, transform, test_dir, batch_size, classes, device, csv_out):
    model.to(device).eval()
    ds = FlatImageDataset(root=test_dir, transform=transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    ids, preds = [], []
    with torch.no_grad():
        for imgs, fnames in loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            idxs = logits.argmax(dim=1).cpu().numpy()
            for i, fname in zip(idxs, fnames):
                ids.append(fname)
                preds.append(classes[i])

    df = pd.DataFrame({"ID": ids, "CLASS": preds})
    df.to_csv(csv_out, index=False)
    print(f"[✓] Submission CSV saved to {csv_out}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Device & configs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    system_cfg = SystemConfig()
    training_cfg = TrainingConfig()

    # Load model
    model = CustomCNN(in_channels=3, num_classes=training_cfg.num_classes)
    checkpoint = torch.load(args.model_path, map_location='cpu')
    last_epoch = checkpoint["epoch"]
    model.load_state_dict(checkpoint["model_state_dict"])

    # Validation loader
    _, valid_loader = get_data(
        batch_size=args.batch_size,
        data_root=args.data_root,
        img_size=training_cfg.img_size,
        num_workers=training_cfg.num_workers,
        data_augmentation=False,
        run_just_a_subset=False,
    )

    # 1) Confusion matrix
    cm_path = os.path.join(args.output_dir, "confusion_matrix.png")
    evaluate_confusion_matrix(
        model, valid_loader, valid_loader.dataset.classes, device, cm_path
    )

    # 2) Submission CSV on Test set
    test_dir = os.path.join(args.data_root, "Test")
    submit_csv = os.path.join(args.output_dir, "submission.csv")
    mean, std = get_mean_std(args.data_root,training_cfg.img_size)
    transform = image_common_transforms(training_cfg.img_size, mean, std)
    generate_submission(
        model,
        transform,
        test_dir,
        args.batch_size,
        valid_loader.dataset.classes,
        device,
        submit_csv,
    )


if __name__ == "__main__":
    main()