#!/usr/bin/env python3
"""
SCRIPT 2: THERMAL-ONLY TRAINING (ViT)
======================================

Trains a Vision Transformer model on Thermal images only.
Compatible with ablation_study.py, extended_metrics.py, grad_cam_visualization.py
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import timm
import random
import numpy as np
import hashlib

# Set random seeds
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# =====================
# Configuration
# =====================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 16
NUM_EPOCHS = 10
LR = 1e-4
WEIGHT_DECAY = 1e-4
# Augmentation and regularization
AUG_PROB = 0.6
DROP_RATE = 0.5
SAVE_BEST_AFTER_EPOCH = 3

# Paths - CHANGE THESE TO YOUR PATIENT-LEVEL DIRECTORIES
DATA_DIR = Path.home() / "CompVis" / "Dataset" / "data"
THERMAL_DIR = DATA_DIR / "thermal"  # Use organized Thermal split directories
CHECKPOINT_DIR = Path.home() / "CompVis" / "DFU_MMT" / "logs" / "checkpoints_thermal_only"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Device: {DEVICE}")
print(f"Thermal Data Directory: {THERMAL_DIR}")
print(f"Checkpoint Directory: {CHECKPOINT_DIR}\n")

# =====================
# Dataset
# =====================
class ThermalDataset(Dataset):
    """Thermal-only dataset"""
    
    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir = Path(data_dir) / split
        self.transform = transform
        
        self.image_paths = []
        self.labels = []
        
        image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
        
        # Load healthy images (label = 0)
        healthy_dir = self.data_dir / 'healthy'
        if healthy_dir.exists():
            for img_path in healthy_dir.rglob('*'):
                if img_path.suffix.lower() in image_exts:
                    self.image_paths.append(img_path)
                    self.labels.append(0)
        
        # Load ulcer images (label = 1)
        ulcer_dir = self.data_dir / 'ulcer'
        if ulcer_dir.exists():
            for img_path in ulcer_dir.rglob('*'):
                if img_path.suffix.lower() in image_exts:
                    self.image_paths.append(img_path)
                    self.labels.append(1)
        
        print(f"  {split.upper()}: {len(self.image_paths)} images "
              f"({self.labels.count(0)} healthy, {self.labels.count(1)} ulcer)")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.long)

# =====================
# Transforms
# =====================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(30),
    transforms.RandomApply([transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.8, 1.2))], p=AUG_PROB),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))], p=AUG_PROB),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# =====================
# Create Datasets & Dataloaders
# =====================
print("Loading datasets...")
train_dataset = ThermalDataset(THERMAL_DIR, 'train', transform=train_transform)
val_dataset = ThermalDataset(THERMAL_DIR, 'val', transform=val_test_transform)
test_dataset = ThermalDataset(THERMAL_DIR, 'test', transform=val_test_transform)

# Leakage check (SHA256) to ensure no exact-image duplicates across splits
def compute_sha256(path, block_size=65536):
    h = hashlib.sha256()
    try:
        with open(path, 'rb') as f:
            for block in iter(lambda: f.read(block_size), b''):
                h.update(block)
        return h.hexdigest()
    except Exception:
        return None

def check_split_hash_leakage_modality(train_ds, val_ds, test_ds, max_samples=5):
    def hashes_for_list(paths):
        d = {}
        for p in paths:
            h = compute_sha256(p)
            if h:
                d.setdefault(h, []).append(str(p))
        return d

    print('\n🔎 Checking for exact-image leakage across thermal splits (SHA256) ...')
    train_paths = [p for p in train_ds.image_paths]
    val_paths = [p for p in val_ds.image_paths]
    test_paths = [p for p in test_ds.image_paths]

    h_train = set(hashes_for_list(train_paths).keys())
    h_val = set(hashes_for_list(val_paths).keys())
    h_test = set(hashes_for_list(test_paths).keys())

    overlap_tv = h_train & h_val
    overlap_tt = h_train & h_test
    overlap_vt = h_val & h_test

    print(f"  Overlaps - train/val: {len(overlap_tv)}, train/test: {len(overlap_tt)}, val/test: {len(overlap_vt)}")
    if len(overlap_tv) + len(overlap_tt) + len(overlap_vt) > 0:
        print('  ⚠️  Found exact duplicates across thermal splits - aborting')
        raise RuntimeError('Image leakage detected across thermal splits')
    else:
        print('  ✅ No exact-image leakage detected for thermal modality')

check_split_hash_leakage_modality(train_dataset, val_dataset, test_dataset)

# Build WeightedRandomSampler for thermal training
from collections import Counter
train_labels = train_dataset.labels
if len(train_labels) > 0:
    counts = Counter(train_labels)
    class_counts = [counts.get(0,0), counts.get(1,0)]
    sample_weights = [1.0 / class_counts[label] if class_counts[label] > 0 else 0.0 for label in train_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4, pin_memory=True)
else:
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# =====================
# Model
# =====================
class ThermalOnlyModel(nn.Module):
    """Vision Transformer for Thermal-only classification"""
    
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        # Load pretrained ViT
        self.backbone = timm.create_model('vit_base_patch16_224', pretrained=pretrained, num_classes=num_classes)
        # insert dropout before final head if present
        if hasattr(self.backbone, 'head'):
            head = self.backbone.head
            try:
                in_features = head.in_features
            except Exception:
                in_features = head.weight.shape[1]
            self.backbone.head = nn.Sequential(nn.Dropout(p=DROP_RATE), nn.Linear(in_features, num_classes))
    
    def forward(self, x):
        return self.backbone(x)

print("Building model...")
model = ThermalOnlyModel(num_classes=2, pretrained=True).to(DEVICE)
print(f"✓ Model loaded: Vision Transformer (ViT-Base)\n")

# =====================
# Loss & Optimizer
# =====================
# class-weighted loss
train_counts = Counter(train_dataset.labels)
class_counts = [train_counts.get(0,0), train_counts.get(1,0)]
total = sum(class_counts) if sum(class_counts)>0 else 1
class_weights = torch.tensor([total / c if c>0 else 0.0 for c in class_counts], dtype=torch.float)
criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# =====================
# Training Loop
# =====================
print("="*70)
print("TRAINING THERMAL-ONLY MODEL (ViT)")
print("="*70)

best_val_f1 = 0.0
history = {'train_loss': [], 'train_acc': [], 'train_f1': [],
           'val_loss': [], 'val_acc': [], 'val_f1': []}

for epoch in range(1, NUM_EPOCHS + 1):
    # ===== TRAINING =====
    model.train()
    train_loss = 0.0
    train_preds = []
    train_labels = []
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [Train]")
    for images, labels in pbar:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Metrics
        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        train_preds.extend(predicted.cpu().numpy())
        train_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    train_loss /= len(train_loader)
    train_acc = accuracy_score(train_labels, train_preds)
    train_f1 = f1_score(train_labels, train_preds, average='binary')
    
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['train_f1'].append(train_f1)
    
    # ===== VALIDATION =====
    model.eval()
    val_loss = 0.0
    val_preds = []
    val_labels = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [Val]  ")
        for images, labels in pbar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_preds.extend(predicted.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())
    
    val_loss /= len(val_loader)
    val_acc = accuracy_score(val_labels, val_preds)
    val_f1 = f1_score(val_labels, val_preds, average='binary')
    
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    history['val_f1'].append(val_f1)
    
    # Print epoch results
    print(f"[Epoch {epoch}/{NUM_EPOCHS}] "
          f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f} | "
          f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
    
    # Prepare checkpoint dict (only saved when it's the best)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_f1': val_f1,
        'history': history
    }

    # Only keep the best model on disk. Delete previous best when a new best is found.
    if epoch >= SAVE_BEST_AFTER_EPOCH and val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_path = CHECKPOINT_DIR / 'best_model.pt'
        try:
            if best_path.exists():
                best_path.unlink()
        except Exception:
            pass
        torch.save(checkpoint, best_path)
        print(f"  💾 Saved BEST model (Val F1: {val_f1:.4f})")

print("\n" + "="*70)
print(f"TRAINING COMPLETE - Best Val F1: {best_val_f1:.4f}")
print("="*70)

# =====================
# Test Evaluation
# =====================
print("\nEvaluating on test set...")
model.eval()
test_loss = 0.0
test_preds = []
test_labels = []
test_probs = []

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Testing"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        test_loss += loss.item()
        probs = torch.softmax(outputs, dim=1)[:, 1]  # Probability of class 1
        _, predicted = torch.max(outputs, 1)
        
        test_preds.extend(predicted.cpu().numpy())
        test_labels.extend(labels.cpu().numpy())
        test_probs.extend(probs.cpu().numpy())

test_loss /= len(test_loader)
test_acc = accuracy_score(test_labels, test_preds)
test_f1 = f1_score(test_labels, test_preds, average='binary')

print("\n" + "="*70)
print("TEST RESULTS (THERMAL-ONLY)")
print("="*70)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Acc:  {test_acc:.4f}")
print(f"Test F1:   {test_f1:.4f}")
print("="*70)

# Save test results
torch.save({
    'test_preds': test_preds,
    'test_labels': test_labels,
    'test_probs': test_probs,
    'test_acc': test_acc,
    'test_f1': test_f1,
    'test_loss': test_loss
}, CHECKPOINT_DIR / 'test_results.pt')

print(f"\n✅ Training complete!")
print(f"Best model saved to: {CHECKPOINT_DIR / 'best_model.pt'}")
print(f"Test results saved to: {CHECKPOINT_DIR / 'test_results.pt'}")
