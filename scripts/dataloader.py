# scripts/dataloader.py
# UPDATED FOR CLEAN ANONYMOUS DATASET

from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import json

# =====================
# UPDATED DATASET CLASS
# =====================

class DFUPairedDataset(Dataset):
    """
    Paired RGB + Thermal DFU dataset with clean organization.
    
    Expects dataset structure created by organize_clean_dataset.py:
    data_dir/rgb/{train,val,test}/healthy|ulcer/*.jpg
    data_dir/thermal/{train,val,test}/healthy|ulcer/*.png
    
    Images are renamed anonymously (h_XXXXX, u_XXXXX, th_XXXXX, tu_XXXXX)
    to prevent model from cheating on filename patterns.
    """
    
    def __init__(self, data_dir, split="train", transform_rgb=None, transform_thermal=None):
        self.rgb_dir = Path(data_dir) / "rgb" / split
        self.thermal_dir = Path(data_dir) / "thermal" / split
        self.transform_rgb = transform_rgb
        self.transform_thermal = transform_thermal
        
        self.rgb_paths = []
        self.thermal_paths = []
        self.labels = []
        
        # Load both modalities with matching indices
        for cls, label in [("healthy", 0), ("ulcer", 1)]:
            rgb_cls_dir = self.rgb_dir / cls
            thermal_cls_dir = self.thermal_dir / cls
            
            # Get all images in each directory
            rgb_cls_paths = sorted(list(rgb_cls_dir.glob("*.*"))) if rgb_cls_dir.exists() else []
            thermal_cls_paths = sorted(list(thermal_cls_dir.glob("*.*"))) if thermal_cls_dir.exists() else []
            
            # Handle missing modality gracefully
            if len(rgb_cls_paths) == 0 and len(thermal_cls_paths) == 0:
                print(f"⚠️  Warning: No {cls} images found in {split}")
                continue
            
            # If one modality is missing, use only available one
            if len(rgb_cls_paths) == 0:
                print(f"⚠️  Warning: No RGB {cls} images - using thermal only")
                self.thermal_paths.extend(thermal_cls_paths)
                self.rgb_paths.extend([None] * len(thermal_cls_paths))  # Placeholder
                self.labels.extend([label] * len(thermal_cls_paths))
            elif len(thermal_cls_paths) == 0:
                print(f"⚠️  Warning: No thermal {cls} images - using RGB only")
                self.rgb_paths.extend(rgb_cls_paths)
                self.thermal_paths.extend([None] * len(rgb_cls_paths))  # Placeholder
                self.labels.extend([label] * len(rgb_cls_paths))
            else:
                # Both modalities available - pair them
                # Take minimum length to ensure both exist
                min_len = min(len(rgb_cls_paths), len(thermal_cls_paths))
                rgb_cls_paths = rgb_cls_paths[:min_len]
                thermal_cls_paths = thermal_cls_paths[:min_len]
                
                self.rgb_paths.extend(rgb_cls_paths)
                self.thermal_paths.extend(thermal_cls_paths)
                self.labels.extend([label] * min_len)
        
        # Verify pairing
        assert len(self.rgb_paths) == len(self.thermal_paths) == len(self.labels), \
            f"Dataset pairing failed: RGB={len(self.rgb_paths)}, Thermal={len(self.thermal_paths)}, Labels={len(self.labels)}"
        
        print(f"✓ Loaded {split} split: {len(self.labels)} total samples")
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # Load RGB (with fallback if missing)
        if self.rgb_paths[idx] is not None:
            rgb_img = Image.open(self.rgb_paths[idx]).convert("RGB")
        else:
            # Create placeholder if missing
            rgb_img = Image.new('RGB', (224, 224), color=0)
        
        # Load Thermal (with fallback if missing)
        if self.thermal_paths[idx] is not None:
            thermal_img = Image.open(self.thermal_paths[idx]).convert("RGB")
        else:
            # Create placeholder if missing
            thermal_img = Image.new('RGB', (224, 224), color=0)
        
        # Apply transforms
        if self.transform_rgb:
            rgb_img = self.transform_rgb(rgb_img)
        
        if self.transform_thermal:
            thermal_img = self.transform_thermal(thermal_img)
        
        label = torch.tensor(self.labels[idx], dtype=torch.float)
        
        return rgb_img, thermal_img, label


class SingleModalityWrapper(torch.utils.data.Dataset):
    """
    Extract single modality from paired dataset for training individual models.
    Usage: wrap DFUPairedDataset to get only RGB or thermal images.
    """
    
    def __init__(self, paired_dataset, modality='rgb'):
        self.paired_dataset = paired_dataset
        self.modality = modality
        assert modality in ('rgb', 'thermal'), f"modality must be 'rgb' or 'thermal', got {modality}"
    
    def __len__(self):
        return len(self.paired_dataset)
    
    def __getitem__(self, idx):
        rgb_img, thermal_img, label = self.paired_dataset[idx]
        
        if self.modality == 'rgb':
            return rgb_img, label
        else:  # thermal
            return thermal_img, label


# =====================
# TRANSFORMS
# =====================

def get_transforms(modality="rgb", augment=True):
    """
    Get transforms for RGB or Thermal images.
    
    Args:
        modality: "rgb" or "thermal"
        augment: whether to apply augmentation
    
    Returns:
        torchvision.transforms.Compose object
    """
    
    if modality == "rgb":
        if augment:
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
        else:
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
    
    elif modality == "thermal":
        if augment:
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5]
                ),
            ])
        else:
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5]
                ),
            ])
    
    else:
        raise ValueError(f"Unknown modality: {modality}")


# =====================
# CONVENIENCE LOADERS
# =====================

def get_dataloaders(data_dir, batch_size=12, num_workers=4, augment=True):
    """
    Get DataLoaders for train/val/test splits.
    
    Args:
        data_dir: path to data/ directory
        batch_size: batch size
        num_workers: number of workers for DataLoader
        augment: whether to apply augmentation
    
    Returns:
        dict of DataLoaders: {'train': ..., 'val': ..., 'test': ...}
        dict of Datasets: {'train': ..., 'val': ..., 'test': ...}
    """
    
    data_dir = Path(data_dir)
    
    rgb_transform = get_transforms("rgb", augment=augment)
    thermal_transform = get_transforms("thermal", augment=augment)
    
    datasets = {}
    dataloaders = {}
    
    for split in ["train", "val", "test"]:
        dataset = DFUPairedDataset(
            data_dir=data_dir,
            split=split,
            transform_rgb=rgb_transform if augment or split == "train" else get_transforms("rgb", augment=False),
            transform_thermal=thermal_transform if augment or split == "train" else get_transforms("thermal", augment=False)
        )
        
        datasets[split] = dataset
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),  # Only shuffle training set
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(split == "train")  # Drop incomplete batches in training
        )
    
    return dataloaders, datasets


def get_single_modality_loaders(data_dir, modality='rgb', batch_size=12, num_workers=4, augment=True):
    """
    Get DataLoaders for single modality (RGB only or Thermal only).
    
    Args:
        data_dir: path to data/ directory
        modality: "rgb" or "thermal"
        batch_size: batch size
        num_workers: number of workers
        augment: whether to apply augmentation
    
    Returns:
        dict of DataLoaders for each split
    """
    
    data_dir = Path(data_dir)
    
    # First get paired datasets
    dataloaders_paired, datasets_paired = get_dataloaders(
        data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        augment=augment
    )
    
    # Wrap with single modality extractor
    dataloaders = {}
    for split in ["train", "val", "test"]:
        wrapped_dataset = SingleModalityWrapper(datasets_paired[split], modality=modality)
        dataloaders[split] = DataLoader(
            wrapped_dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(split == "train")
        )
    
    return dataloaders


# =====================
# DATASET STATISTICS
# =====================

def print_dataset_statistics(data_dir):
    """Print statistics about the dataset"""
    print("\n" + "="*70)
    print("DATASET STATISTICS")
    print("="*70)
    
    data_dir = Path(data_dir)
    
    for split in ["train", "val", "test"]:
        print(f"\n{split.upper()} SET:")
        for modality in ["rgb", "thermal"]:
            healthy_count = len(list((data_dir / modality / split / "healthy").glob("*.*")))
            ulcer_count = len(list((data_dir / modality / split / "ulcer").glob("*.*")))
            total = healthy_count + ulcer_count
            
            if total > 0:
                healthy_pct = 100 * healthy_count / total
                ulcer_pct = 100 * ulcer_count / total
                print(f"  {modality.upper():8s}: {healthy_count:4d} healthy ({healthy_pct:5.1f}%), "
                      f"{ulcer_count:4d} ulcer ({ulcer_pct:5.1f}%), Total: {total:4d}")
            else:
                print(f"  {modality.upper():8s}: No images found")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    # Test the dataloader
    data_dir = Path.home() / "CompVis" / "Dataset" / "data"
    
    if data_dir.exists():
        # Print statistics
        print_dataset_statistics(data_dir)
        
        # Load dataloaders
        dataloaders, datasets = get_dataloaders(data_dir, batch_size=4, num_workers=0)
        
        # Test iteration
        print("\nTesting dataloader iteration:")
        for split in ["train", "val", "test"]:
            if len(datasets[split]) > 0:
                rgb, thermal, label = datasets[split][0]
                print(f"  {split}: RGB shape={rgb.shape}, Thermal shape={thermal.shape}, Label={label}")
    else:
        print(f"Dataset not found at {data_dir}")
        print("Please run organize_clean_dataset.py first")