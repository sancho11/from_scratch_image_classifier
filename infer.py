"""
infer.py

For classifying images with a trained model, computing a confusion matrix
on the validation set, and generating a CSV submission file for test images.

Usage:
  # Compute confusion matrix and CSV submission:
  python infer.py --model_path /path/to/checkpoint.pt

  # Also classify a single image:
  python infer.py --model_path /path/to/checkpoint.pt \
                  --image_path  /path/to/image.jpg
"""

import os
import argparse
import glob

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from common import (
    CustomCNN,
    TrainingConfig,
    get_mean_std,
    image_common_transforms,
)

def load_model(checkpoint_path: str, num_classes: int, device: torch.device):
    """Instantiate model and load state dict from checkpoint_path."""
    model = CustomCNN(num_classes=num_classes)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model

def classify_single_image(model, image_path, img_size, mean, std, device):
    """Run a single-image classification and print the predicted class."""
    transform = image_common_transforms(img_size, mean, std)
    img = Image.open(image_path).convert("RGB")
    inp = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(inp)
        pred = out.argmax(dim=1).item()

    # derive class names from validation folder
    # assume same classes as in train/valid
    # grab via ImageFolder on valid folder
    # (fallback to train if needed)
    return pred

def main():
    parser = argparse.ArgumentParser(description="Inference script")
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="Path to the model checkpoint (.pt)"
    )
    parser.add_argument(
        "--val_data_dir", type=str, default='./dataset/Valid',
        help="Path to validation data root (contains class subfolders)"
    )
    parser.add_argument(
        "--image_path", type=str,  required=True,
        help="Classify this single image and print the result"
    )
    args = parser.parse_args()

    # Load config defaults
    cfg = TrainingConfig()
    data_root = cfg.data_root
    img_size = tuple(cfg.img_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths
    val_dir = args.val_data_dir or os.path.join(data_root, "Valid")
    # Load model
    model = CustomCNN(in_channels=3, num_classes=cfg.num_classes)
    checkpoint = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    mean, std = get_mean_std(cfg.data_root,cfg.img_size)

    transform = image_common_transforms(img_size, mean, std)
    # derive class names from validation folder
    val_ds = datasets.ImageFolder(root=val_dir, transform=transform)
    class_names = val_ds.classes

    img = Image.open(args.image_path).convert("RGB")
    inp = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(inp)
        pred_idx = out.argmax(dim=1).item()
    print(f"Predicted class for {args.image_path}: {class_names[pred_idx]}")

if __name__ == "__main__":
    main()