

import os
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch

import config


#  Augmentation pipelines
# Training gets random transforms for variety.
# Val/Test get ONLY resize + normalize for fair evaluation.

def get_train_transforms():
    return transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD),
    ])

def get_val_transforms():
    return transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD),
    ])


# Data loading function
def get_dataloaders():
    """
    Reads train/val/test folders using ImageFolder.
    Returns three DataLoaders + the class-to-index mapping.
    """

    # ImageFolder scans subfolders and assigns labels automatically
    train_ds = datasets.ImageFolder(config.TRAIN_DIR, transform=get_train_transforms())
    val_ds   = datasets.ImageFolder(config.VAL_DIR,   transform=get_val_transforms())
    test_ds  = datasets.ImageFolder(config.TEST_DIR,  transform=get_val_transforms())

    class_to_idx = train_ds.class_to_idx
    num_classes  = len(class_to_idx)

    print(f"Classes found ({num_classes}): {list(class_to_idx.keys())}")
    print(f"Split sizes → Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}\n")
    
    targets      = np.array(train_ds.targets)           
    class_counts = np.bincount(targets)                
    class_weights = 1.0 / class_counts                  

    
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    print("Training class distribution:")
    for i, (count, weight) in enumerate(zip(class_counts, class_weights)):
        print(f"  {idx_to_class[i]:<30} {count:>5} images")
    print()

    
    sample_weights = class_weights[targets]
    sampler = WeightedRandomSampler(
        weights     = torch.tensor(sample_weights, dtype=torch.float),
        num_samples = len(train_ds),
        replacement = True,
    )

    # DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size  = config.BATCH_SIZE,
        sampler     = sampler,      # weighted sampler replaces shuffle=True
        num_workers = 2,            # set to 0 if you get weird Windows errors
        pin_memory  = True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = config.BATCH_SIZE,
        shuffle     = False,
        num_workers = 2,
        pin_memory  = True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size  = config.BATCH_SIZE,
        shuffle     = False,
        num_workers = 2,
        pin_memory  = True,
    )

    return train_loader, val_loader, test_loader, class_to_idx