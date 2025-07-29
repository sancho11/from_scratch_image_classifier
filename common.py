import os, sys
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# ---------------------- Transforms & Data Loading ---------------------- #

def image_preprocess_transforms(img_size: Tuple[int,int]):
    """
    Create basic image preprocessing transforms.

    Parameters:
        img_size (tuple[int, int] or int): Desired output size (height, width) for images.

    Returns:
        torchvision.transforms.Compose: Transform pipeline for resizing and tensor conversion.
    """
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
    ])


def image_common_transforms(img_size: Tuple[int,int], mean: Tuple[float,float,float], std: Tuple[float,float,float]):
    """
    Build transform pipeline with normalization.

    Parameters:
        img_size (tuple[int, int]): Target size for images.
        mean (tuple[float, float, float]): Channel means for normalization.
        std (tuple[float, float, float]): Channel standard deviations for normalization.

    Returns:
        torchvision.transforms.Compose: Full pipeline including resize, tensor conversion, and normalization.
    """
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def get_mean_std(data_root: str, img_size: Tuple[int,int], num_workers: int = 4) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate per-channel mean and standard deviation for images in a folder.

    Parameters:
        data_root (str): Path to dataset root (contains subfolders for classes).
        img_size (tuple[int, int]): Size for resizing during computation.
        num_workers (int): Number of DataLoader workers.

    Returns:
        mean (torch.Tensor): Channel-wise mean.
        std (torch.Tensor): Channel-wise standard deviation.
    """
    transform = image_preprocess_transforms(img_size)
    if not os.path.isdir(data_root):
            sys.exit(
                f"\nERROR: Dataset directory not found at:\n"
                f"  {data_root}\n\n"
                "Please download the dataset archive and unzip it into the above directory before running this script.\n"
                "For more information on how to get dataset please check `get_dataset.md` \n"
            )
    loader = DataLoader(
        datasets.ImageFolder(root=data_root, transform=transform),
        batch_size=16, shuffle=False, num_workers=num_workers
    )
        
    sum_ = torch.zeros(3)
    sum_sq = torch.zeros(3)
    n = 0
    for imgs, _ in loader:
        sum_ += imgs.mean(dim=[0,2,3])
        sum_sq += (imgs**2).mean(dim=[0,2,3])
        n += 1
    mean = sum_ / n
    var = (sum_sq / n) - mean**2
    std = torch.sqrt(var)
    return mean.tolist(), std.tolist()


def data_loader(data_root: str, transform, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    """
    Create a DataLoader for all images in a folder.

    Parameters:
        data_root (str): Root directory of dataset.
        transform (callable): Transform pipeline to apply.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle the dataset.
        num_workers (int): Number of subprocesses for data loading.

    Returns:
        DataLoader: PyTorch DataLoader instance.
    """
    dataset = datasets.ImageFolder(root=data_root, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def subset_data_loader(data_root: str, transform, batch_size: int, shuffle: bool, num_workers: int, subset_size: float = 0.05) -> DataLoader:
    """
    Create a DataLoader using a fixed subset of the dataset for quick iterations.

    Parameters:
        data_root (str): Root directory of dataset.
        transform (callable): Transform pipeline to apply.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle the subset.
        num_workers (int): Number of subprocesses for data loading.
        subset_size (float): Fraction of the full dataset to use.

    Returns:
        DataLoader: PyTorch DataLoader of the subset.
    """
    dataset = datasets.ImageFolder(root=data_root, transform=transform)
    total = len(dataset)
    indices = np.linspace(0, total-1, int(total*subset_size), dtype=int)
    subset = Subset(dataset, indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def get_data(
    batch_size: int,
    data_root: str,
    img_size: Tuple[int,int],
    num_workers: int,
    data_augmentation: bool,
    run_just_a_subset: bool
) -> Tuple[DataLoader, DataLoader]:
    """
    Prepare training and validation DataLoaders with preprocessing, normalization, and optional augmentation.

    Parameters:
        batch_size (int): Batch size for both loaders.
        data_root (str): Root directory containing 'Train' and 'Valid' subfolders.
        img_size (tuple[int, int]): Target image size.
        num_workers (int): Number of subprocesses for data loading.
        data_augmentation (bool): Whether to apply data augmentation on training data.
        run_just_a_subset (bool): If True, use a small subset for faster iteration.

    Returns:
        train_loader (DataLoader): DataLoader for training dataset.
        valid_loader (DataLoader): DataLoader for validation dataset.
    """
    train_dir = os.path.join(data_root, 'Train')
    valid_dir = os.path.join(data_root, 'Valid')

    mean, std = get_mean_std(train_dir, img_size, num_workers)
    common_transforms = image_common_transforms(img_size, mean, std)

    if data_augmentation:
        train_transforms = transforms.Compose([
            transforms.Resize(int(min(img_size)*1.15)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=(0,359), translate=(0.1,0.3), scale=(0.4,1.05)),
            transforms.ColorJitter(brightness=.5, hue=.3),
            transforms.RandomEqualize(),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        train_transforms = common_transforms

    loader_fn = subset_data_loader if run_just_a_subset else data_loader
    train_loader = loader_fn(train_dir, train_transforms, batch_size, True, num_workers)
    valid_loader = loader_fn(valid_dir, common_transforms, batch_size, False, num_workers)
    return train_loader, valid_loader

# ---------------------- Configuration Dataclasses ---------------------- #

@dataclass
class SystemConfig:
    """
    Configuration for system-level reproducibility and performance.

    Attributes:
        seed (int): Random seed for all RNGs.
        cudnn_benchmark_enabled (bool): Enable CuDNN benchmark for speed.
        cudnn_deterministic (bool): Force deterministic CuDNN algorithms.
    """
    seed: int = 21
    cudnn_benchmark_enabled: bool = True
    cudnn_deterministic: bool = True

@dataclass
class TrainingConfig:
    """
    Hyperparameters and I/O paths for model training.

    Attributes:
        num_classes (int): Number of target classes.
        batch_size (int): Samples per batch.
        img_size (Tuple[int,int]): Input image dimensions.
        epochs_count (int): Total training epochs.
        decay_rate (float): Learning rate decay factor.
        init_learning_rate (float): Starting learning rate.
        data_root (str): Base path to dataset folder.
        num_workers (int): DataLoader worker count.
        device (str): Compute device ('cpu' or 'cuda').
        save_best_model_name (str): Filename for best model checkpoint.
        save_last_model_name (str): Filename for last model checkpoint.
        root_log_dir (str): Base directory for TensorBoard logs.
        root_checkpoint_dir (str): Base directory for saved checkpoints.
        half_precision (bool): Enables or disables using half precision
    """
    num_classes: int = 3
    batch_size: int = 16
    img_size: Tuple[int,int] = (238,238)
    epochs_count: int = 60
    decay_rate: float = 0.01
    init_learning_rate: float = 1e-3
    data_root: str = './dataset'
    num_workers: int = 4
    device: str = 'cuda'
    save_best_model_name: str = 'best_model.pt'
    save_last_model_name: str = 'last_model.pt'
    root_log_dir: str = 'Logs_Checkpoints/Model_logs'
    root_checkpoint_dir: str = 'Logs_Checkpoints/Model_checkpoints'
    half_precision: bool = False

# ---------------------- System Setup & Persistence ---------------------- #

def setup_system(config: SystemConfig):
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = config.cudnn_benchmark_enabled
        torch.backends.cudnn.deterministic = config.cudnn_deterministic


def save_model(model: nn.Module, epoch:int, device: str, model_dir: str, model_file_name: str):
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, model_file_name)
    if device == 'cuda': model.to('cpu')

    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict()
    }
    ,model_path)


    if device == 'cuda': model.to('cuda')


def setup_log_directory(training_config: TrainingConfig) -> Tuple[TrainingConfig, str]:
    root_logs = training_config.root_log_dir
    if os.path.isdir(root_logs):
        versions = [int(name.replace('version_','')) for name in os.listdir(root_logs)
                    if name.startswith('version_') and name.replace('version_','').isdigit()]
        next_ver = max(versions, default=-1) + 1
        version_name = f'version_{next_ver}'
    else:
        version_name = 'version_0'
    training_config.log_dir = os.path.join(root_logs, version_name)
    training_config.checkpoint_dir = os.path.join(training_config.root_checkpoint_dir, version_name)
    os.makedirs(training_config.log_dir, exist_ok=True)
    os.makedirs(training_config.checkpoint_dir, exist_ok=True)
    return training_config, version_name


# ---------------------- Model Definition ---------------------- #

class CustomCNN(nn.Module):
    """
    Residual-style CNN for image classification.

    Args:
        in_channels (int): Number of input channels (e.g., 3 for RGB).
        num_classes (int): Number of output classes.
        drop_spatial (float): Dropout probability after conv3 block.
        drop_dense (float): Dropout probability before final linear layer.
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 3,
                 drop_spatial: float = 0.1, drop_dense: float = 0.2):
        super().__init__()
        # Block 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=32, kernel_size=3, bias=False),
            nn.BatchNorm2d(32), 
            nn.ReLU(inplace=True)
        )
        self.res1a = nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.res1b = nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,padding=1),
            nn.BatchNorm2d(32), 
            nn.ReLU(inplace=True)
        )
        # Block 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,bias=False), 
            nn.BatchNorm2d(64), 
            nn.ReLU(inplace=True)
        )
        self.res2a = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,padding=1), 
            nn.BatchNorm2d(64), 
            nn.ReLU(inplace=True)
        )
        self.res2b = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,padding=1), 
            nn.BatchNorm2d(64), 
            nn.ReLU(inplace=True)
        )
        # Block 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,bias=False), 
            nn.BatchNorm2d(128), 
            nn.ReLU(inplace=True)
        )
        self.res3a = nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,padding=1), 
            nn.BatchNorm2d(128), 
            nn.ReLU(inplace=True)
        )
        self.res3b = nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,padding=1), 
            nn.BatchNorm2d(128), 
            nn.ReLU(inplace=True)
        )
        # Block 4
        self.res4a = nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.res4b = nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        #Block 5
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=512,kernel_size=3,bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=drop_spatial),
        )

        #Block 6
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=512,out_channels=64,kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )


        # Head
        self.pool = nn.MaxPool2d(2)
        self.fc_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=64*2*2, out_features=128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop_dense),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=64, out_features=num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the CNN.

        Args:
            x (torch.Tensor): Batch of input images.

        Returns:
            torch.Tensor: Logits for each class.
        """
        # Block 1 with skip
        x = self.conv1(x)#238->236
        skip = x
        x = self.res1a(x)
        x = self.res1b(x) + skip
        x = self.pool(x)#236->118

        # Block 2 with skip
        x = self.conv2(x)#118->116
        skip = x
        x = self.res2a(x)
        x = self.res2b(x) + skip
        x = self.pool(x)#116->58

        # Block 3 with skip
        x = self.conv3(x)#58->56
        skip = x
        x = self.res3a(x)
        x = self.res3b(x) + skip
        x = self.pool(x)#56->28

        # Block 4 with skip
        skip = x
        x = self.res4a(x)
        x = self.res4b(x) + skip
        x = self.pool(x)#28->14

        #Block 5 without skip
        x = self.conv5(x)#14->12
        x = self.pool(x) #12->6

        #Block 6 without skip
        x = self.conv6(x)#6->4
        x = self.pool(x) #4->2
        return self.fc_head(x)