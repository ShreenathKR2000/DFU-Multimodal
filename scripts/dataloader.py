# scripts/dataloader.py
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# =====================
# Dataset
# =====================
class DFUPairedDataset(Dataset):
    """
    Paired RGB + Thermal DFU dataset.
    Expects:
        data_dir/rgb_standardized/{train,val,test}/healthy|ulcer/*.jpg
        data_dir/thermal_standardized/{train,val,test}/healthy|ulcer/*.png
    """

    def __init__(self, data_dir, split="train", transform_rgb=None, transform_thermal=None):
        self.rgb_dir = Path(data_dir) / "rgb_standardized" / split
        self.thermal_dir = Path(data_dir) / "thermal_standardized" / split
        self.transform_rgb = transform_rgb
        self.transform_thermal = transform_thermal

        self.rgb_paths = []
        self.thermal_paths = []
        self.labels = []

        for cls, label in [("healthy", 0), ("ulcer", 1)]:
            rgb_cls_paths = sorted(list((self.rgb_dir / cls).glob("*.*")))
            thermal_cls_paths = sorted(list((self.thermal_dir / cls).glob("*.*")))

            # --- Pairing logic: take min length ---
            min_len = min(len(rgb_cls_paths), len(thermal_cls_paths))
            rgb_cls_paths = rgb_cls_paths[:min_len]
            thermal_cls_paths = thermal_cls_paths[:min_len]

            self.rgb_paths.extend(rgb_cls_paths)
            self.thermal_paths.extend(thermal_cls_paths)
            self.labels.extend([label] * min_len)

        assert len(self.rgb_paths) == len(self.thermal_paths) == len(self.labels), "Pairing failed!"

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        rgb_img = Image.open(self.rgb_paths[idx]).convert("RGB")
        thermal_img = Image.open(self.thermal_paths[idx]).convert("RGB")  # 3-channel thermal

        if self.transform_rgb:
            rgb_img = self.transform_rgb(rgb_img)
        if self.transform_thermal:
            thermal_img = self.transform_thermal(thermal_img)

        label = torch.tensor(self.labels[idx], dtype=torch.float)
        return rgb_img, thermal_img, label

# =====================
# Transforms
# =====================
def get_transforms(modality="rgb"):
    if modality == "rgb":
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15)
        ])
    elif modality == "thermal":
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15)
        ])
    else:
        raise ValueError(f"Unknown modality: {modality}")

# =====================
# Convenience loader
# =====================
def get_dataloaders(data_dir, batch_size=12, num_workers=4):
    rgb_transform = get_transforms("rgb")
    thermal_transform = get_transforms("thermal")

    datasets = {split: DFUPairedDataset(
        data_dir=Path(data_dir),
        split=split,
        transform_rgb=rgb_transform,
        transform_thermal=thermal_transform
    ) for split in ["train","val","test"]}

    dataloaders = {split: DataLoader(
        datasets[split],
        batch_size=batch_size,
        shuffle=(split=="train"),
        num_workers=num_workers,
        pin_memory=True
    ) for split in ["train","val","test"]}

    return dataloaders, datasets
