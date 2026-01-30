#!/usr/bin/env python3
"""
SCRIPT 3: MULTIMODAL FUSION TRAINING (ResNet50 + ViT)
======================================================

Trains a multimodal fusion model combining RGB (ResNet50) and Thermal (ViT).
Compatible with ablation_study.py, extended_metrics.py, grad_cam_visualization.py

This is your main model for comparison with single-modality baselines.
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
BATCH_SIZE = 6  # Smaller batch for multimodal (more memory)
NUM_EPOCHS = 10
LR = 1e-4
WEIGHT_DECAY = 1e-4
# Augmentation and regularization
AUG_PROB = 0.6
DROP_RATE = 0.5
SAVE_BEST_AFTER_EPOCH = 3

# Paths - CHANGE THESE TO YOUR PATIENT-LEVEL DIRECTORIES
DATA_DIR = Path.home() / "CompVis" / "Dataset" / "data"
RGB_DIR = DATA_DIR / "rgb"
THERMAL_DIR = DATA_DIR / "thermal"
CHECKPOINT_DIR = Path.home() / "CompVis" / "DFU_MMT" / "logs" / "checkpoints_multimodal"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Device: {DEVICE}")
print(f"RGB Data Directory: {RGB_DIR}")
print(f"Thermal Data Directory: {THERMAL_DIR}")
print(f"Checkpoint Directory: {CHECKPOINT_DIR}\n")

# =====================
# Dataset for Paired Data
# =====================
class MultimodalDataset(Dataset):
    """
    Paired RGB + Thermal dataset
    
    Note: Since RGB and Thermal are from different sources, this creates
    pseudo-pairing based on matching indices within each class.
    """
    
    def __init__(self, rgb_dir, thermal_dir, split='train', 
                 transform_rgb=None, transform_thermal=None):
        self.rgb_dir = Path(rgb_dir) / split
        self.thermal_dir = Path(thermal_dir) / split
        self.transform_rgb = transform_rgb
        self.transform_thermal = transform_thermal
        
        # Load RGB images
        self.rgb_healthy = []
        self.rgb_ulcer = []
        
        # Load Thermal images
        self.thermal_healthy = []
        self.thermal_ulcer = []
        
        image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
        
        # RGB healthy
        rgb_healthy_dir = self.rgb_dir / 'healthy'
        if rgb_healthy_dir.exists():
            self.rgb_healthy = sorted([p for p in rgb_healthy_dir.rglob('*') 
                                      if p.suffix.lower() in image_exts])
        
        # RGB ulcer
        rgb_ulcer_dir = self.rgb_dir / 'ulcer'
        if rgb_ulcer_dir.exists():
            self.rgb_ulcer = sorted([p for p in rgb_ulcer_dir.rglob('*') 
                                    if p.suffix.lower() in image_exts])
        
        # Thermal healthy
        thermal_healthy_dir = self.thermal_dir / 'healthy'
        if thermal_healthy_dir.exists():
            self.thermal_healthy = sorted([p for p in thermal_healthy_dir.rglob('*') 
                                          if p.suffix.lower() in image_exts])
        
        # Thermal ulcer
        thermal_ulcer_dir = self.thermal_dir / 'ulcer'
        if thermal_ulcer_dir.exists():
            self.thermal_ulcer = sorted([p for p in thermal_ulcer_dir.rglob('*') 
                                        if p.suffix.lower() in image_exts])
        
        # Create paired indices
        # Strategy: cycle through smaller dataset to match larger one
        self.pairs = []
        
        # Pair healthy images (only if both modalities have healthy examples)
        if self.rgb_healthy and self.thermal_healthy:
            num_healthy_pairs = max(len(self.rgb_healthy), len(self.thermal_healthy))
            for i in range(num_healthy_pairs):
                rgb_idx = i % len(self.rgb_healthy)
                thermal_idx = i % len(self.thermal_healthy)
                self.pairs.append((self.rgb_healthy[rgb_idx], self.thermal_healthy[thermal_idx], 0))
        else:
            if not self.rgb_healthy:
                print('  Warning: no RGB healthy images found; skipping healthy pairing')
            if not self.thermal_healthy:
                print('  Warning: no Thermal healthy images found; skipping healthy pairing')

        # Pair ulcer images (only if both modalities have ulcer examples)
        if self.rgb_ulcer and self.thermal_ulcer:
            num_ulcer_pairs = max(len(self.rgb_ulcer), len(self.thermal_ulcer))
            for i in range(num_ulcer_pairs):
                rgb_idx = i % len(self.rgb_ulcer)
                thermal_idx = i % len(self.thermal_ulcer)
                self.pairs.append((self.rgb_ulcer[rgb_idx], self.thermal_ulcer[thermal_idx], 1))
        else:
            if not self.rgb_ulcer:
                print('  Warning: no RGB ulcer images found; skipping ulcer pairing')
            if not self.thermal_ulcer:
                print('  Warning: no Thermal ulcer images found; skipping ulcer pairing')
        
        # Shuffle pairs
        random.shuffle(self.pairs)
        
        healthy_count = sum(1 for _, _, label in self.pairs if label == 0)
        ulcer_count = sum(1 for _, _, label in self.pairs if label == 1)
        
        print(f"  {split.upper()}: {len(self.pairs)} pairs "
              f"({healthy_count} healthy, {ulcer_count} ulcer)")
        print(f"    RGB: {len(self.rgb_healthy)} healthy, {len(self.rgb_ulcer)} ulcer")
        print(f"    Thermal: {len(self.thermal_healthy)} healthy, {len(self.thermal_ulcer)} ulcer")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        rgb_path, thermal_path, label = self.pairs[idx]
        
        # Load images
        rgb_image = Image.open(rgb_path).convert('RGB')
        thermal_image = Image.open(thermal_path).convert('RGB')
        
        # Apply transforms
        if self.transform_rgb:
            rgb_image = self.transform_rgb(rgb_image)
        
        if self.transform_thermal:
            thermal_image = self.transform_thermal(thermal_image)
        
        return rgb_image, thermal_image, torch.tensor(label, dtype=torch.long)

# =====================
# Transforms
# =====================
# RGB transforms
rgb_train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(30),
    transforms.RandomApply([transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3)], p=AUG_PROB),
    transforms.RandomApply([transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.8, 1.2))], p=AUG_PROB),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

rgb_val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Thermal transforms
thermal_train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(30),
    transforms.RandomApply([transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.8, 1.2))], p=AUG_PROB),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

thermal_val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# =====================
# Create Datasets & Dataloaders
# =====================
print("Loading datasets...")
train_dataset = MultimodalDataset(
    RGB_DIR, THERMAL_DIR, 'train',
    transform_rgb=rgb_train_transform,
    transform_thermal=thermal_train_transform
)

val_dataset = MultimodalDataset(
    RGB_DIR, THERMAL_DIR, 'val',
    transform_rgb=rgb_val_test_transform,
    transform_thermal=thermal_val_test_transform
)

test_dataset = MultimodalDataset(
    RGB_DIR, THERMAL_DIR, 'test',
    transform_rgb=rgb_val_test_transform,
    transform_thermal=thermal_val_test_transform
)

# Leakage check (SHA256) across RGB and Thermal splits
def compute_sha256(path, block_size=65536):
    h = hashlib.sha256()
    try:
        with open(path, 'rb') as f:
            for block in iter(lambda: f.read(block_size), b''):
                h.update(block)
        return h.hexdigest()
    except Exception:
        return None

def check_multimodal_leakage(train_ds, val_ds, test_ds):
    print('\n🔎 Checking for exact-image leakage across multimodal splits...')
    def collect_paths(ds):
        rgb = [str(p) for p,_,_ in ds.pairs]
        thermal = [str(t) for _,t,_ in ds.pairs]
        return rgb, thermal

    tr_rgb, tr_th = collect_paths(train_ds)
    va_rgb, va_th = collect_paths(val_ds)
    te_rgb, te_th = collect_paths(test_ds)

    def overlaps(a,b):
        ha = {compute_sha256(p) for p in a}
        hb = {compute_sha256(p) for p in b}
        return len(ha & hb)

    print(f"  RGB overlaps tr/val: {overlaps(tr_rgb, va_rgb)}, tr/test: {overlaps(tr_rgb, te_rgb)}, val/test: {overlaps(va_rgb, te_rgb)}")
    print(f"  Thermal overlaps tr/val: {overlaps(tr_th, va_th)}, tr/test: {overlaps(tr_th, te_th)}, val/test: {overlaps(va_th, te_th)}")

    # If any overlap > 0, raise
    if overlaps(tr_rgb, va_rgb) + overlaps(tr_rgb, te_rgb) + overlaps(va_rgb, te_rgb) + overlaps(tr_th, va_th) + overlaps(tr_th, te_th) + overlaps(va_th, te_th) > 0:
        raise RuntimeError('Exact-image leakage detected across multimodal splits')
    else:
        print('  ✅ No exact-image leakage detected for multimodal splits')

check_multimodal_leakage(train_dataset, val_dataset, test_dataset)

# Build WeightedRandomSampler for paired training
from collections import Counter
train_labels = [label for _,_,label in train_dataset.pairs]
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
# Multimodal Fusion Model
# =====================
class MultimodalFusionModel(nn.Module):
    """
    Late fusion of ResNet50 (RGB) and ViT (Thermal)
    """
    
    def __init__(self, num_classes=2, dropout=0.5):
        super().__init__()
        
        # RGB branch: ResNet50
        self.rgb_branch = torch.hub.load('pytorch/vision:v0.13.1', 'resnet50', pretrained=True)
        rgb_feat_dim = self.rgb_branch.fc.in_features
        self.rgb_branch.fc = nn.Identity()  # Remove final FC layer
        
        # Thermal branch: ViT
        self.thermal_branch = timm.create_model('vit_base_patch16_224', 
                                               pretrained=True, 
                                               num_classes=0)  # No classifier
        thermal_feat_dim = 768  # ViT-Base feature dimension
        
        # Fusion classifier
        self.fusion = nn.Sequential(
            nn.Linear(rgb_feat_dim + thermal_feat_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, rgb, thermal):
        # Extract features from both modalities
        rgb_features = self.rgb_branch(rgb)        # (B, 2048)
        thermal_features = self.thermal_branch(thermal)  # (B, 768)
        
        # Concatenate features
        fused_features = torch.cat([rgb_features, thermal_features], dim=1)  # (B, 2816)
        
        # Classify
        output = self.fusion(fused_features)  # (B, 2)
        
        return output

print("Building model...")
model = MultimodalFusionModel(num_classes=2, dropout=DROP_RATE).to(DEVICE)
print(f"✓ Model loaded: ResNet50 (RGB) + ViT (Thermal) with late fusion\n")

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}\n")

# =====================
# Loss & Optimizer
# =====================
# class-weighted loss
train_counts = Counter(train_labels)
class_counts = [train_counts.get(0,0), train_counts.get(1,0)]
total = sum(class_counts) if sum(class_counts)>0 else 1
class_weights = torch.tensor([total / c if c>0 else 0.0 for c in class_counts], dtype=torch.float)
criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# =====================
# Training Loop
# =====================
print("="*70)
print("TRAINING MULTIMODAL FUSION MODEL (ResNet50 + ViT)")
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
    for rgb_images, thermal_images, labels in pbar:
        rgb_images = rgb_images.to(DEVICE)
        thermal_images = thermal_images.to(DEVICE)
        labels = labels.to(DEVICE)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(rgb_images, thermal_images)
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
        for rgb_images, thermal_images, labels in pbar:
            rgb_images = rgb_images.to(DEVICE)
            thermal_images = thermal_images.to(DEVICE)
            labels = labels.to(DEVICE)
            
            outputs = model(rgb_images, thermal_images)
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
    for rgb_images, thermal_images, labels in tqdm(test_loader, desc="Testing"):
        rgb_images = rgb_images.to(DEVICE)
        thermal_images = thermal_images.to(DEVICE)
        labels = labels.to(DEVICE)
        
        outputs = model(rgb_images, thermal_images)
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
print("TEST RESULTS (MULTIMODAL FUSION)")
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
