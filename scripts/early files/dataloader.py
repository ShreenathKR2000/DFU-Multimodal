#!/usr/bin/env python3
"""
PyTorch DataLoader Setup for Multimodal DFU Classification
Ready to use with standardized 224×224 images
"""

import os
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

class DFUDataset(Dataset):
    """
    Custom PyTorch Dataset for DFU multimodal images
    Loads paired RGB and Thermal images with shared labels
    """
    
    def __init__(self, rgb_dir, thermal_dir, split='train', transform_rgb=None, transform_thermal=None):
        """
        Args:
            rgb_dir: Path to RGB images directory
            thermal_dir: Path to Thermal images directory
            split: 'train', 'val', or 'test'
            transform_rgb: PyTorch transforms for RGB images
            transform_thermal: PyTorch transforms for Thermal images
        """
        self.rgb_dir = Path(rgb_dir) / split
        self.thermal_dir = Path(thermal_dir) / split
        self.transform_rgb = transform_rgb
        self.transform_thermal = transform_thermal
        
        # Collect image paths
        self.rgb_paths = []
        self.thermal_paths = []
        self.labels = []
        
        # Image extensions
        self.image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
        
        # Load RGB images
        rgb_healthy_dir = self.rgb_dir / 'healthy'
        rgb_ulcer_dir = self.rgb_dir / 'ulcer'
        
        if rgb_healthy_dir.exists():
            healthy_paths = self._get_image_paths(rgb_healthy_dir)
            self.rgb_paths.extend(healthy_paths)
            self.labels.extend([0] * len(healthy_paths))
        
        if rgb_ulcer_dir.exists():
            ulcer_paths = self._get_image_paths(rgb_ulcer_dir)
            self.rgb_paths.extend(ulcer_paths)
            self.labels.extend([1] * len(ulcer_paths))
        
        # Load Thermal images (same number and order of labels)
        thermal_healthy_dir = self.thermal_dir / 'healthy'
        thermal_ulcer_dir = self.thermal_dir / 'ulcer'
        
        if thermal_healthy_dir.exists():
            healthy_paths = self._get_image_paths(thermal_healthy_dir)
            self.thermal_paths.extend(healthy_paths)
        
        if thermal_ulcer_dir.exists():
            ulcer_paths = self._get_image_paths(thermal_ulcer_dir)
            self.thermal_paths.extend(ulcer_paths)
        
        print(f"Loaded {split.upper()} split:")
        print(f"  RGB images:     {len(self.rgb_paths)}")
        print(f"  Thermal images: {len(self.thermal_paths)}")
        print(f"  Labels (0=healthy, 1=ulcer): {self.labels.count(0)} healthy, {self.labels.count(1)} ulcer")
    
    def _get_image_paths(self, directory):
        """Get sorted list of image paths"""
        images = []
        for ext in self.image_exts:
            images.extend(directory.glob(f'*{ext}'))
            images.extend(directory.glob(f'*{ext.upper()}'))
        return sorted(images)
    
    def __len__(self):
        return min(len(self.rgb_paths), len(self.thermal_paths))
    
    def __getitem__(self, idx):
        """
        Returns:
            rgb_img: Tensor of RGB image
            thermal_img: Tensor of Thermal image
            label: 0 (healthy) or 1 (ulcer)
        """
        # Load RGB image
        rgb_path = self.rgb_paths[idx]
        rgb_img = Image.open(rgb_path).convert('RGB')
        
        if self.transform_rgb:
            rgb_img = self.transform_rgb(rgb_img)
        
        # Load Thermal image
        thermal_path = self.thermal_paths[idx]
        thermal_img = Image.open(thermal_path).convert('RGB')  # Already converted during standardization
        
        if self.transform_thermal:
            thermal_img = self.transform_thermal(thermal_img)
        
        # Get label
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return rgb_img, thermal_img, label


def get_transforms(modality='rgb'):
    """
    Get appropriate transforms for each modality
    
    Args:
        modality: 'rgb' or 'thermal'
    
    Returns:
        transforms.Compose object
    """
    
    if modality.lower() == 'rgb':
        return transforms.Compose([
            transforms.Resize((224, 224), interpolation=Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet statistics
                std=[0.229, 0.224, 0.225]
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(15),
        ])
    
    elif modality.lower() == 'thermal':
        return transforms.Compose([
            transforms.Resize((224, 224), interpolation=Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],  # Thermal-specific
                std=[0.5, 0.5, 0.5]
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(15),
        ])
    
    else:
        raise ValueError(f"Unknown modality: {modality}")


# def get_dataloaders(data_dir, batch_size=32, num_workers=4):
#     """
#     Create DataLoaders for training, validation, and testing
#     
#     Args:
#         data_dir: Path to data directory (should contain rgb_standardized/ and thermal_standardized/)
#         batch_size: Batch size for training
#         num_workers: Number of worker processes for data loading
#     
#     Returns:
#         dict with 'train', 'val', 'test' DataLoaders
#     """
#     
#     data_dir = Path(data_dir)
#     rgb_dir = data_dir / 'rgb_standardized'
#     thermal_dir = data_dir / 'thermal_standardized'
#     
#     # Verify directories exist
#     if not rgb_dir.exists() or not thermal_dir.exists():
#         raise FileNotFoundError(f"Standardized data directories not found!")
#     
#     # Get transforms
#     rgb_transforms = get_transforms('rgb')
#     thermal_transforms = get_transforms('thermal')
#     
#     # Create datasets
#     print("="*70)
#     print("CREATING DATALOADERS")
#     print("="*70 + "\n")
#     
#     datasets = {}
#     dataloaders = {}
#     
#     for split in ['train', 'val', 'test']:
#         print(f"Loading {split.upper()} split...")
#         datasets[split] = DFUDataset(
#             rgb_dir=rgb_dir,
#             thermal_dir=thermal_dir,
#             split=split,
#             transform_rgb=rgb_transforms,
#             transform_thermal=thermal_transforms
#         )
#         print()
#         
#         # Adjust batch size for smaller splits
#         split_batch_size = batch_size if split == 'train' else max(16, batch_size // 2)
#         
#         dataloaders[split] = DataLoader(
#             datasets[split],
#             batch_size=split_batch_size,
#             shuffle=(split == 'train'),
#             num_workers=num_workers,
#             pin_memory=True
#         )
#     
#     return dataloaders, datasets

def get_dataloaders(data_dir, batch_size=32, num_workers=4, modality="both"):
    """
    Create DataLoaders for training, validation, and testing.
    
    Args:
        data_dir: Path to data directory (should contain rgb_standardized/ and thermal_standardized/)
        batch_size: Batch size for training
        num_workers: Number of worker processes for data loading
        modality: 'rgb', 'thermal', or 'both'
    
    Returns:
        dict of DataLoaders for 'train', 'val', 'test'
    """
    data_dir = Path(data_dir)
    rgb_dir = data_dir / 'rgb_standardized'
    thermal_dir = data_dir / 'thermal_standardized'

    if modality not in {"rgb", "thermal", "both"}:
        raise ValueError(f"Unknown modality: {modality}")

    # Prepare transforms
    rgb_transforms = get_transforms('rgb') if modality in {"rgb", "both"} else None
    thermal_transforms = get_transforms('thermal') if modality in {"thermal", "both"} else None

    datasets = {}
    dataloaders = {}

    print("="*70)
    print("CREATING DATALOADERS")
    print("="*70 + "\n")

    for split in ['train', 'val', 'test']:
        print(f"Loading {split.upper()} split...")

        datasets[split] = DFUDataset(
            rgb_dir=rgb_dir,
            thermal_dir=thermal_dir,
            split=split,
            transform_rgb=rgb_transforms,
            transform_thermal=thermal_transforms
        )

        split_batch_size = batch_size if split == 'train' else max(16, batch_size // 2)
        dataloaders[split] = DataLoader(
            datasets[split],
            batch_size=split_batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            pin_memory=True
        )
        print()

    return dataloaders, datasets


def test_dataloaders():
    """Test the dataloaders to ensure they work correctly"""
    
    print("\n" + "="*70)
    print("TESTING DATALOADERS")
    print("="*70)
    
    # Get base directory
    base_dir = Path.home() / "DFU_MMT" / "data"
    
    if not base_dir.exists():
        print(f"❌ Data directory not found: {base_dir}")
        return False
    
    try:
        # Create dataloaders
        dataloaders, datasets = get_dataloaders(base_dir, batch_size=4)
        
        print(f"\n✓ Successfully created dataloaders!\n")
        
        # Test loading a batch
        print("Testing batch loading from TRAIN split...")
        train_loader = dataloaders['train']
        
        rgb_batch, thermal_batch, label_batch = next(iter(train_loader))
        
        print(f"✓ Batch loaded successfully!")
        print(f"\n  RGB Batch Shape:     {rgb_batch.shape}")
        print(f"  Thermal Batch Shape: {thermal_batch.shape}")
        print(f"  Label Batch Shape:   {label_batch.shape}")
        print(f"  Labels:              {label_batch}")
        
        # Verify shapes
        assert rgb_batch.shape == (4, 3, 224, 224), "RGB batch shape incorrect!"
        assert thermal_batch.shape == (4, 3, 224, 224), "Thermal batch shape incorrect!"
        assert label_batch.shape == (4,), "Label batch shape incorrect!"
        
        print(f"\n✅ All shapes correct!")
        
        # Print dataset statistics
        print(f"\n" + "="*70)
        print("DATASET STATISTICS")
        print(f"="*70)
        
        for split, dataset in datasets.items():
            labels = dataset.labels
            healthy = labels.count(0)
            ulcer = labels.count(1)
            total = len(labels)
            print(f"{split.upper():5s}: {total:4d} images ({healthy:4d} healthy, {ulcer:4d} ulcer)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating dataloaders: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Test the dataloaders
    success = test_dataloaders()
    
    if success:
        print(f"\n{'='*70}")
        print("✅ DATALOADERS READY FOR TRAINING!")
        print(f"{'='*70}")
        print(f"\nUsage in training code:")
        print(f"""
from dataloader import get_dataloaders

dataloaders, datasets = get_dataloaders(
    data_dir=Path.home() / 'DFU_MMT' / 'data',
    batch_size=32,
    num_workers=4
)

# Training loop
for epoch in range(num_epochs):
    for rgb_batch, thermal_batch, labels in dataloaders['train']:
        # Your training code here
        rgb_batch = rgb_batch.to(device)
        thermal_batch = thermal_batch.to(device)
        labels = labels.to(device)
        
        # Forward pass
        rgb_features = rgb_model(rgb_batch)
        thermal_features = thermal_model(thermal_batch)
        fused = fusion(rgb_features, thermal_features)
        output = classifier(fused)
        
        # Loss & backprop
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        """)
    else:
        print(f"\n{'='*70}")
        print("❌ DATALOADER TEST FAILED")
        print(f"{'='*70}")