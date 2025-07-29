# train.py

import os, sys
import time
import argparse
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torchmetrics import MeanMetric
from torchmetrics.classification import MulticlassAccuracy
from tqdm import tqdm
from torch.amp import GradScaler, autocast
from common import get_data, image_common_transforms, CustomCNN, save_model, TrainingConfig, SystemConfig, setup_system, setup_log_directory

# ---------------------- Training & Validation Loops ---------------------- #

def train_epoch(
    train_config: TrainingConfig,
    model: nn.Module,
    optimizer: optim.Optimizer,
    train_loader: DataLoader,
    epoch_idx: int
) -> Tuple[float, float]:
    """
    Execute one training epoch.

    Args:
        train_config (TrainingConfig): Training hyperparameters.
        model (nn.Module): PyTorch model to train.
        optimizer (optim.Optimizer): Optimizer instance.
        train_loader (DataLoader): Training data loader.
        epoch_idx (int): Current epoch index (1-based).
        total_epochs (int): Total number of epochs.

    Returns:
        epoch_loss (float): Average loss over epoch.
        epoch_acc (float): Average accuracy over epoch.
    """
    model.train()
    loss_metric = MeanMetric()
    device = train_config.device
    acc_metric = MulticlassAccuracy(num_classes=train_config.num_classes).to(device)

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        loss_metric(loss.item(), weight=data.size(0))
        preds = output.argmax(dim=1)
        acc_metric(preds, target)
    return loss_metric.compute().item(), acc_metric.compute().item()


def validate_epoch(
    train_config: TrainingConfig,
    model: nn.Module,
    valid_loader: DataLoader
) -> Tuple[float, float]:
    """
    Execute one validation epoch.

    Args:
        train_config (TrainingConfig): Training hyperparameters.
        model (nn.Module): PyTorch model to evaluate.
        valid_loader (DataLoader): Validation data loader.
        epoch_idx (int): Current epoch index (1-based).
        total_epochs (int): Total number of epochs.

    Returns:
        valid_loss (float): Average validation loss.
        valid_acc (float): Average validation accuracy.
    """
    model.eval()
    acc_metric = MulticlassAccuracy(num_classes=train_config.num_classes).to(device)
    loss_metric = MeanMetric()
    device = train_config.device
    with torch.no_grad():
        for data, target in valid_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss_metric(loss.item(), weight=data.size(0))
            preds = output.argmax(dim=1)
            acc_metric(preds, target)
    return loss_metric.compute().item(), acc_metric.compute().item()

# ---------------------- Main Training Routine ---------------------- #

def main(
    model: nn.Module,
    writer: SummaryWriter,
    use_scheduler: bool,
    system_config: SystemConfig,
    training_config: TrainingConfig,
    data_augmentation: bool,
    run_just_a_subset: bool,
    last_epoch: int = 0
) -> Tuple[List[float], List[float], List[float], List[float]]:
    setup_system(system_config)
    train_loader, valid_loader = get_data(
        training_config.batch_size,
        training_config.data_root,
        training_config.img_size,
        training_config.num_workers,
        data_augmentation,
        run_just_a_subset
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    training_config.device = str(device)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=training_config.init_learning_rate, amsgrad=True)
    scheduler = None
    if use_scheduler:
        lr_fn = lambda epoch: 1 / (1 + training_config.decay_rate * epoch)
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_fn)

    scaler = GradScaler(device="cuda") if training_config.half_precision and device.type == 'cuda' else None

    best_loss = float('inf'); best_acc = 0.0
    train_losses, train_accs, val_losses, val_accs = [], [], [], []

    for epoch in tqdm(range(1+last_epoch, training_config.epochs_count+1+last_epoch), desc='Epochs'):
        # --- Training ---
        model.train()
        train_loss_metric = MeanMetric();
        train_acc_metric = MulticlassAccuracy(num_classes=training_config.num_classes).to(device)
        for data, target in tqdm(train_loader, desc=f'Train Epoch {epoch}', leave=False):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            with autocast(device_type="cuda", enabled=(scaler is not None)):
                output = model(data)
                loss = F.cross_entropy(output, target)
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            train_loss_metric(loss.item(), weight=data.size(0))
            preds = output.argmax(dim=1)
            train_acc_metric(preds, target)
        tr_loss = train_loss_metric.compute().item(); tr_acc = train_acc_metric.compute().item()

        # --- Validation ---
        model.eval()
        val_loss_metric = MeanMetric(); 
        val_acc_metric = MulticlassAccuracy(num_classes=training_config.num_classes).to(device)
        with torch.no_grad():
            for data, target in tqdm(valid_loader, desc=f'Val Epoch {epoch}', leave=False):
                data, target = data.to(device), target.to(device)
                with autocast(device_type="cuda", enabled=(scaler is not None)):
                    output = model(data)
                    loss = F.cross_entropy(output, target)
                val_loss_metric(loss.item(), weight=data.size(0))
                preds = output.argmax(dim=1)
                val_acc_metric(preds, target)
        val_loss = val_loss_metric.compute().item(); val_acc = val_acc_metric.compute().item()

        if scheduler: scheduler.step()
        train_losses.append(tr_loss); train_accs.append(tr_acc)
        val_losses.append(val_loss); val_accs.append(val_acc)
        writer.add_scalar('Loss/Train', tr_loss, epoch)
        writer.add_scalar('Acc/Train', tr_acc, epoch)
        writer.add_scalar('Loss/Val', val_loss, epoch)
        writer.add_scalar('Acc/Val', val_acc, epoch)

        # --- Checkpointing ---
        if val_loss < best_loss:
            best_loss = val_loss
        if val_acc > best_acc:
            best_acc = val_acc
            save_model(model, epoch, training_config.device, training_config.checkpoint_dir, training_config.save_best_model_name)
        if epoch % 5 == 0:
            save_model(model, epoch, training_config.device, training_config.checkpoint_dir, training_config.save_last_model_name)
        print(f"Epoch {epoch:03d}: TrLoss={tr_loss:.4f}, TrAcc={tr_acc:.4f}, ValLoss={val_loss:.4f}, ValAcc={val_acc:.4f}")

    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f}s | Best Loss: {best_loss:.4f}, Best Acc: {best_acc:.4f}")
    return train_losses, train_accs, val_losses, val_accs

# ---------------------- CLI Interface ---------------------- #

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train CustomCNN from scratch')
    parser.add_argument('--data_root', type=str, default='./dataset', help='Root directory of dataset')
    parser.add_argument('--epochs_count', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--img_size', type=int, nargs=2, default=[238,238], help='Image size H W')
    parser.add_argument('--init_learning_rate', type=float, default=1e-3, help='Initial LR')
    parser.add_argument('--decay_rate', type=float, default=0.01, help='LR decay rate')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--use_scheduler', action='store_true', help='Use LR scheduler')
    parser.add_argument('--data_augmentation', action='store_true', help='Apply data augmentation')
    parser.add_argument('--run_just_a_subset', action='store_true', help='Use small subset for quick debug')
    parser.add_argument('--log_root', type=str, default='Logs_Checkpoints/Model_logs', help='TensorBoard log root')
    parser.add_argument('--checkpoint_root', type=str, default='Logs_Checkpoints/Model_checkpoints', help='Checkpoint root')
    parser.add_argument('--half_precision', action='store_true', help='Enable automatic mixed precision training')
    parser.add_argument("--resume",type=str, default=None, help="Path to a checkpoint file to resume training from")
    args = parser.parse_args()

    # Build config
    training_config = TrainingConfig(
        batch_size=args.batch_size,
        img_size=tuple(args.img_size),
        epochs_count=args.epochs_count,
        init_learning_rate=args.init_learning_rate,
        decay_rate=args.decay_rate,
        data_root=args.data_root,
        num_workers=args.num_workers,
        root_log_dir=args.log_root,
        root_checkpoint_dir=args.checkpoint_root,
        half_precision=args.half_precision
    )
    training_config, version = setup_log_directory(training_config)
    writer = SummaryWriter(training_config.log_dir)
    system_config = SystemConfig()

    model = CustomCNN(in_channels=3, num_classes=training_config.num_classes)
    last_epoch=0
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location='cpu')
        last_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["model_state_dict"])
        
    start_time = time.time()
    main(
        model,
        writer,
        use_scheduler=args.use_scheduler,
        system_config=system_config,
        training_config=training_config,
        data_augmentation=args.data_augmentation,
        run_just_a_subset=args.run_just_a_subset,
        last_epoch=last_epoch
    )