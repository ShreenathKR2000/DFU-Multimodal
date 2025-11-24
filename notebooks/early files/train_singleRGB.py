# train_baseline.py
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from PIL import Image

# Add DFU_MMT root to sys.path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from scripts.dataloader import DFUDataset, get_transforms
from models.encoders import EfficientNetEncoder
from models.classifier import Classifier

# =====================
# CONFIGURATION
# =====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 15
NUM_WORKERS = 4
NUM_EPOCHS = 10
LR = 1e-4

# Select modality: "rgb" or "thermal"
MODALITY = "rgb"

# =====================
# TRANSFORMS
# =====================
transforms_mod = get_transforms(MODALITY)

# =====================
# DATASET AND DATALOADER
# =====================
data_dir = project_root / "data"
rgb_dir = data_dir / "rgb_standardized"
thermal_dir = data_dir / "thermal_standardized"

class SingleModalityDFUDataset(DFUDataset):
    def __getitem__(self, idx):
        # Load label
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        # Load RGB or Thermal based on MODALITY
        if MODALITY == "rgb":
            img_path = self.rgb_paths[idx]
            img = Image.open(img_path).convert("RGB")
            if self.transform_rgb:
                img = self.transform_rgb(img)
        elif MODALITY == "thermal":
            img_path = self.thermal_paths[idx]
            img = Image.open(img_path).convert("RGB")
            if self.transform_thermal:
                img = self.transform_thermal(img)
        else:
            raise ValueError(f"Unknown modality: {MODALITY}")

        return img, label

# Create datasets and dataloaders
datasets = {}
dataloaders = {}

for split in ["train", "val", "test"]:
    datasets[split] = SingleModalityDFUDataset(
        rgb_dir=rgb_dir,
        thermal_dir=thermal_dir,
        split=split,
        transform_rgb=transforms_mod if MODALITY=="rgb" else None,
        transform_thermal=transforms_mod if MODALITY=="thermal" else None
    )
    dataloaders[split] = DataLoader(
        datasets[split],
        batch_size=BATCH_SIZE,
        shuffle=(split=="train"),
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

# =====================
# MODEL
# =====================
encoder = EfficientNetEncoder().to(DEVICE)
classifier = Classifier().to(DEVICE)  # uses default feat_dim=1280

criterion = nn.BCELoss()
optimizer = AdamW(list(encoder.parameters()) + list(classifier.parameters()), lr=LR, weight_decay=1e-4)

# =====================
# TRAIN LOOP
# =====================
for epoch in range(NUM_EPOCHS):
    encoder.train()
    classifier.train()
    running_loss = 0
    all_preds, all_labels = [], []

    for inputs, labels in dataloaders["train"]:
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        features = encoder(inputs)
        output = classifier(features).squeeze()
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        all_preds.append(torch.round(output.detach().cpu()))
        all_labels.append(labels.detach().cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    print(f"[Epoch {epoch+1}/{NUM_EPOCHS}] Loss: {running_loss/len(dataloaders['train']):.4f}, Acc: {acc:.4f}, F1: {f1:.4f}")
