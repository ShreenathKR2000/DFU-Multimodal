#!/usr/bin/env python3
"""
Multimodal DFU Classifier: RGB (ResNet50) + Thermal (ViT)
- Late concatenation of features
- MLP classifier with strong dropout
- Modality-specific augmentations
- Label noise + random batch-level intensity variations
- Validation logging
- Grad-CAM hooks ready
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
import timm
import numpy as np
import random
from torchvision import transforms

# -----------------------------
# Project root & scripts
# -----------------------------
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))
from scripts.dataloader import DFUPairedDataset

# -----------------------------
# Config
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 6
NUM_EPOCHS = 15
LR = 1e-4

CHECKPOINT_DIR = project_root / "logs" / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = project_root / "data"

# =====================
# AUGMENTATIONS
# =====================

# RGB strong augmentation
rgb_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.15),
    transforms.RandomRotation(40),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomAffine(degrees=40, translate=(0.15,0.15), scale=(0.7,1.3)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# Thermal augmentation with noise + mask
class ThermalRandomNoise(object):
    def __init__(self, mean=0.0, std=0.05): self.mean, self.std = mean, std
    def __call__(self, tensor): return tensor + torch.randn_like(tensor) * self.std

class ThermalRandomMask(object):
    def __init__(self, mask_prob=0.3, max_size=40): self.mask_prob, self.max_size = mask_prob, max_size
    def __call__(self, tensor):
        if random.random() < self.mask_prob:
            _, h, w = tensor.shape
            mh, mw = random.randint(5,self.max_size), random.randint(5,self.max_size)
            x, y = random.randint(0,w-mw), random.randint(0,h-mh)
            tensor[:, y:y+mh, x:x+mw] = 0
        return tensor

thermal_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomAffine(degrees=20, translate=(0.1,0.1), scale=(0.8,1.2)),
    transforms.ToTensor(),
    ThermalRandomNoise(std=0.1),
    ThermalRandomMask(mask_prob=0.3),
    transforms.Normalize([0.5]*3,[0.5]*3)
])

# =====================
# Datasets & Dataloaders
# =====================
datasets = {split: DFUPairedDataset(
    data_dir=DATA_DIR,
    split=split,
    transform_rgb=rgb_transform,
    transform_thermal=thermal_transform
) for split in ["train","val","test"]}

dataloaders = {split: DataLoader(
    datasets[split],
    batch_size=BATCH_SIZE,
    shuffle=(split=="train"),
    num_workers=4,
    pin_memory=True
) for split in ["train","val","test"]}

# =====================
# Helper: Label noise
# =====================
def add_label_noise(labels, noise_ratio=0.1):
    n = len(labels)
    num_flip = int(noise_ratio * n)
    flip_idx = np.random.choice(n, size=num_flip, replace=False)
    labels[flip_idx] = 1 - labels[flip_idx]
    return labels

# =====================
# Helper: Mixup for paired data
# =====================
def mixup_data(x1, x2, y, alpha=0.4):
    """
    Applies Mixup on paired multimodal data:
    x1: RGB batch
    x2: Thermal batch
    y: labels (0/1)
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x1.size(0)
    index = torch.randperm(batch_size).to(x1.device)

    mixed_x1 = lam * x1 + (1 - lam) * x1[index, :]
    mixed_x2 = lam * x2 + (1 - lam) * x2[index, :]
    y_a, y_b = y, y[index]
    return mixed_x1, mixed_x2, y_a, y_b, lam

# =====================
# Model
# =====================
class MLPFusion(nn.Module):
    def __init__(self, rgb_feat_dim=2048, thermal_feat_dim=768, hidden_dim=512):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(rgb_feat_dim + thermal_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.7),   # stronger dropout to reduce overfitting
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, rgb_feat, thermal_feat):
        fused = torch.cat([rgb_feat, thermal_feat], dim=1)
        return self.classifier(fused)

# Encoders
resnet = torch.hub.load('pytorch/vision:v0.13.1', 'resnet50', pretrained=True)
resnet.fc = nn.Identity()
resnet = resnet.to(DEVICE)

vit = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
vit = vit.to(DEVICE)

fusion_model = MLPFusion().to(DEVICE)

# Loss + optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = AdamW(list(resnet.parameters()) + list(vit.parameters()) + list(fusion_model.parameters()),
                  lr=LR, weight_decay=1e-4)

# --- Mixup loss function ---
def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Computes BCE loss for Mixup labels.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# =====================
# TRAIN LOOP
# =====================
best_val_f1 = 0.0

for epoch in range(1, NUM_EPOCHS+1):
    resnet.train(); vit.train(); fusion_model.train()
    running_loss = 0
    all_preds, all_labels = [], []

    for rgb_batch, th_batch, labels in dataloaders["train"]:
        rgb_batch, th_batch = rgb_batch.to(DEVICE), th_batch.to(DEVICE)
        # Random intensity scaling
        if random.random() < 0.2: rgb_batch *= random.uniform(0.5,1.0)
        if random.random() < 0.2: th_batch *= random.uniform(0.5,1.0)

        # Add label noise
        labels = add_label_noise(labels.cpu().numpy(), noise_ratio=0.1)
        labels = torch.tensor(labels).float().to(DEVICE)

        # --- Apply Mixup ---
        rgb_batch, th_batch, y_a, y_b, lam = mixup_data(rgb_batch, th_batch, labels, alpha=0.4)

        optimizer.zero_grad()
        rgb_feat = resnet(rgb_batch)
        th_feat = vit(th_batch)
        output = fusion_model(rgb_feat, th_feat).squeeze()
        loss = mixup_criterion(criterion, output, y_a, y_b, lam)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        all_preds.append(torch.round(torch.sigmoid(output)).cpu())
        all_labels.append(labels.cpu())

    # TRAIN METRICS
    all_preds = torch.cat(all_preds).detach().cpu().numpy()  # detach here
    all_labels = torch.cat(all_labels).detach().cpu().numpy()  # detach here

    # For Mixup, use threshold 0.5 to get binary predictions
    all_preds_bin = (all_preds > 0.5).astype(int)

    train_loss = running_loss / len(dataloaders["train"])
    train_acc = accuracy_score(all_labels, all_preds_bin)
    train_f1 = f1_score(all_labels, all_preds_bin)
    train_loss = running_loss / len(dataloaders["train"])

    # Validation
    resnet.eval(); vit.eval(); fusion_model.eval()
    val_loss = 0
    val_preds, val_labels = [], []
    with torch.no_grad():
        for rgb_batch, th_batch, labels in dataloaders["val"]:
            rgb_batch, th_batch, labels = rgb_batch.to(DEVICE), th_batch.to(DEVICE), labels.float().to(DEVICE)
            rgb_feat = resnet(rgb_batch)
            th_feat = vit(th_batch)
            output = fusion_model(rgb_feat, th_feat).squeeze()
            loss = criterion(output, labels)
            val_loss += loss.item()
            val_preds.append(torch.round(torch.sigmoid(output)).cpu())
            val_labels.append(labels.cpu())

    val_preds = torch.cat(val_preds).detach().cpu().numpy()
    val_labels = torch.cat(val_labels).detach().cpu().numpy()
    val_preds_bin = (val_preds > 0.5).astype(int)

    val_acc = accuracy_score(val_labels, val_preds_bin)
    val_f1 = f1_score(val_labels, val_preds_bin)
    val_loss /= len(dataloaders["val"])

    print(f"[Epoch {epoch}/{NUM_EPOCHS}] "
          f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f} | "
          f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")

    # Checkpoint
    checkpoint_path = CHECKPOINT_DIR / f"epoch_{epoch}.pt"
    torch.save({
        "epoch": epoch,
        "resnet_state": resnet.state_dict(),
        "vit_state": vit.state_dict(),
        "fusion_state": fusion_model.state_dict(),
        "optimizer_state": optimizer.state_dict()
    }, checkpoint_path)

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save({
            "epoch": epoch,
            "resnet_state": resnet.state_dict(),
            "vit_state": vit.state_dict(),
            "fusion_state": fusion_model.state_dict(),
            "optimizer_state": optimizer.state_dict()
        }, CHECKPOINT_DIR / f"best_valF1_{val_f1:.4f}.pt")
        print(f"💾 Saved NEW BEST model -> {CHECKPOINT_DIR / f'best_valF1_{val_f1:.4f}.pt'}")

# =====================
# Test evaluation
# =====================
resnet.eval(); vit.eval(); fusion_model.eval()
test_loss = 0; test_preds, test_labels = [], []
with torch.no_grad():
    for rgb_batch, th_batch, labels in dataloaders["test"]:
        rgb_batch, th_batch, labels = rgb_batch.to(DEVICE), th_batch.to(DEVICE), labels.float().to(DEVICE)
        rgb_feat = resnet(rgb_batch)
        th_feat = vit(th_batch)
        output = fusion_model(rgb_feat, th_feat).squeeze()
        loss = criterion(output, labels)
        test_loss += loss.item()
        test_preds.append(torch.round(torch.sigmoid(output)).cpu())
        test_labels.append(labels.cpu())

test_preds = torch.cat(test_preds)
test_labels = torch.cat(test_labels)
test_acc = accuracy_score(test_labels, test_preds)
test_f1 = f1_score(test_labels, test_preds)
test_loss /= len(dataloaders["test"])
print(f"\nTest Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, F1: {test_f1:.4f}")
