#!/usr/bin/env python3
"""
EXTENDED MEDICAL METRICS WITH FLEXIBLE CHECKPOINT LOADING
==========================================================

Handles checkpoint mismatches automatically:
- Different FC layer sizes (1 vs 2 outputs)
- Different naming conventions (backbone vs resnet)
- Loads only compatible layers
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, precision_recall_curve, auc,
    matthews_corrcoef, cohen_kappa_score
)
import matplotlib.pyplot as plt
from tqdm import tqdm
import timm
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms
import random

# Project paths
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))


# =====================
# FLEXIBLE CHECKPOINT LOADER
# =====================

def load_checkpoint_flexible(model, checkpoint_path, device='cuda'):
    """Load checkpoint with automatic handling of architecture mismatches"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get('model_state_dict', {})
    
    if not state_dict:
        return False
    
    model_state = model.state_dict()
    
    # Fix naming: backbone -> resnet or vit
    # Training uses 'backbone.layer4.0.conv1.weight' -> need 'resnet.layer4.0.conv1.weight'
    # Training uses 'backbone.head.1.weight' -> need 'vit.head.1.weight'
    fixed_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('backbone.'):
            # Remove 'backbone.' prefix
            new_key = key[9:]  # len('backbone.') = 9
            # Add appropriate prefix based on model type
            if hasattr(model, 'resnet'):
                new_key = 'resnet.' + new_key
            elif hasattr(model, 'vit'):
                new_key = 'vit.' + new_key
            fixed_state_dict[new_key] = value
        else:
            fixed_state_dict[key] = value
    
    # Load with size mismatch handling
    loaded_keys = []
    skipped_keys = []
    for key, value in fixed_state_dict.items():
        if key not in model_state:
            continue
        
        # Check for size mismatch
        if value.shape != model_state[key].shape:
            # Skip FC/head layer if size mismatch (will use random init)
            if 'fc' in key or 'head' in key or 'classifier' in key:
                skipped_keys.append(f"{key}: {value.shape} vs {model_state[key].shape}")
                continue
            else:
                continue
        
        # Load matching keys
        model_state[key] = value
        loaded_keys.append(key)
    
    # Load the state dict
    model.load_state_dict(model_state, strict=False)
    print(f"  Loaded {len(loaded_keys)} layers from checkpoint")
    if skipped_keys:
        print(f"  Skipped {len(skipped_keys)} layers due to shape mismatch")
    return True


# =====================
# SINGLE MODALITY DATASETS
# =====================

class RGBDataset(Dataset):
    """RGB-only dataset matching training structure"""
    def __init__(self, data_dir, split='test', transform=None):
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
        
        print(f"  RGB {split.upper()}: {len(self.image_paths)} images "
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


class ThermalDataset(Dataset):
    """Thermal-only dataset matching training structure"""
    def __init__(self, data_dir, split='test', transform=None):
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
        
        print(f"  Thermal {split.upper()}: {len(self.image_paths)} images "
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


class MultimodalDataset(Dataset):
    """Paired RGB + Thermal dataset matching training structure"""
    def __init__(self, rgb_dir, thermal_dir, split='test', transform_rgb=None, transform_thermal=None):
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
        self.pairs = []
        
        # Pair healthy images
        if self.rgb_healthy and self.thermal_healthy:
            num_healthy_pairs = max(len(self.rgb_healthy), len(self.thermal_healthy))
            for i in range(num_healthy_pairs):
                rgb_idx = i % len(self.rgb_healthy)
                thermal_idx = i % len(self.thermal_healthy)
                self.pairs.append((self.rgb_healthy[rgb_idx], self.thermal_healthy[thermal_idx], 0))
        
        # Pair ulcer images
        if self.rgb_ulcer and self.thermal_ulcer:
            num_ulcer_pairs = max(len(self.rgb_ulcer), len(self.thermal_ulcer))
            for i in range(num_ulcer_pairs):
                rgb_idx = i % len(self.rgb_ulcer)
                thermal_idx = i % len(self.thermal_ulcer)
                self.pairs.append((self.rgb_ulcer[rgb_idx], self.thermal_ulcer[thermal_idx], 1))
        
        # Shuffle pairs
        random.seed(42)
        random.shuffle(self.pairs)
        
        healthy_count = sum(1 for _, _, label in self.pairs if label == 0)
        ulcer_count = sum(1 for _, _, label in self.pairs if label == 1)
        
        print(f"  Multimodal {split.upper()}: {len(self.pairs)} pairs "
              f"({healthy_count} healthy, {ulcer_count} ulcer)")
    
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
# CONFIGURATION
# =====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_DIR = project_root / "logs"
DATA_DIR = Path.home() / "CompVis" / "Dataset" / "data"
RGB_DIR = DATA_DIR / "rgb"
THERMAL_DIR = DATA_DIR / "thermal"
OUTPUT_DIR = CHECKPOINT_DIR / "extended_metrics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Evaluation transforms (ensure consistent tensor shapes for DataLoader)
rgb_eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

thermal_eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


# =====================
# MODEL DEFINITIONS
# =====================

class RGBOnlyModel(nn.Module):
    def __init__(self, num_classes=2, drop_rate=0.5):
        super().__init__()
        self.resnet = torch.hub.load('pytorch/vision:v0.13.1', 'resnet50', pretrained=True)
        self.resnet.fc = nn.Sequential(
            nn.Dropout(p=drop_rate),
            nn.Linear(2048, num_classes)
        )
    
    def forward(self, x):
        return self.resnet(x)


class ThermalOnlyModel(nn.Module):
    """ViT for Thermal-only"""
    def __init__(self, num_classes=2, drop_rate=0.5):
        super().__init__()
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)
        # Add dropout before head if present
        if hasattr(self.vit, 'head'):
            head = self.vit.head
            try:
                in_features = head.in_features
            except Exception:
                in_features = head.weight.shape[1]
            self.vit.head = nn.Sequential(nn.Dropout(p=drop_rate), nn.Linear(in_features, num_classes))
    
    def forward(self, x):
        return self.vit(x)


class MLPFusion(nn.Module):
    def __init__(self, rgb_feat_dim=2048, thermal_feat_dim=768, hidden_dim=512, num_classes=2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(rgb_feat_dim + thermal_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, rgb_feat, thermal_feat):
        fused = torch.cat([rgb_feat, thermal_feat], dim=1)
        return self.classifier(fused)


class MultimodalFusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = torch.hub.load('pytorch/vision:v0.13.1', 'resnet50', pretrained=True)
        self.resnet.fc = nn.Identity()
        
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
        
        self.fusion = MLPFusion(rgb_feat_dim=2048, thermal_feat_dim=768, hidden_dim=512)
    
    def forward(self, rgb, thermal):
        rgb_feat = self.resnet(rgb)
        thermal_feat = self.vit(thermal)
        output = self.fusion(rgb_feat, thermal_feat)
        return output


# =====================
# METRICS CALCULATOR
# =====================

class MedicalMetricsCalculator:
    """Calculate comprehensive medical imaging metrics"""
    
    def __init__(self, y_true, y_pred, y_probs=None):
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.y_probs = np.array(y_probs) if y_probs is not None else None
        
        self.metrics = {}
        self.calculate_all_metrics()
    
    def calculate_all_metrics(self):
        self.confusion_matrix_metrics()
        self.classification_metrics()
        self.sensitivity_specificity()
        self.roc_auc_metrics()
        self.additional_metrics()
    
    def confusion_matrix_metrics(self):
        cm = confusion_matrix(self.y_true, self.y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        self.metrics['confusion_matrix'] = cm
        self.metrics['tn'] = tn
        self.metrics['fp'] = fp
        self.metrics['fn'] = fn
        self.metrics['tp'] = tp
    
    def classification_metrics(self):
        self.metrics['accuracy'] = accuracy_score(self.y_true, self.y_pred)
        self.metrics['precision'] = precision_score(self.y_true, self.y_pred, zero_division=0)
        self.metrics['recall'] = recall_score(self.y_true, self.y_pred, zero_division=0)
        self.metrics['f1'] = f1_score(self.y_true, self.y_pred, zero_division=0)
        
        self.metrics['classification_report'] = classification_report(
            self.y_true, self.y_pred,
            target_names=['Healthy', 'Ulcer'],
            zero_division=0
        )
    
    def sensitivity_specificity(self):
        """Medical imaging critical metrics"""
        tn = self.metrics['tn']
        fp = self.metrics['fp']
        fn = self.metrics['fn']
        tp = self.metrics['tp']
        
        self.metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        self.metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        self.metrics['ppv'] = tp / (tp + fp) if (tp + fp) > 0 else 0
        self.metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0
        self.metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0
        self.metrics['fnr'] = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    def roc_auc_metrics(self):
        if self.y_probs is None:
            self.metrics['auc_roc'] = None
            self.metrics['auc_pr'] = None
            return
        
        self.metrics['auc_roc'] = roc_auc_score(self.y_true, self.y_probs)
        precision, recall, _ = precision_recall_curve(self.y_true, self.y_probs)
        self.metrics['auc_pr'] = auc(recall, precision)
    
    def additional_metrics(self):
        self.metrics['mcc'] = matthews_corrcoef(self.y_true, self.y_pred)
        self.metrics['kappa'] = cohen_kappa_score(self.y_true, self.y_pred)
        
        sens = self.metrics['sensitivity']
        spec = self.metrics['specificity']
        self.metrics['balanced_accuracy'] = (sens + spec) / 2
        self.metrics['f_harmonic'] = 2 * (sens * spec) / (sens + spec) if (sens + spec) > 0 else 0
    
    def get_all_metrics(self):
        return self.metrics
    
    def print_report(self, model_name="Model"):
        print("\n" + "="*70)
        print(f"EXTENDED MEDICAL METRICS: {model_name}")
        print("="*70)
        
        print("\n🎯 BASIC CLASSIFICATION METRICS:")
        print(f"  Accuracy:       {self.metrics['accuracy']:.4f}")
        print(f"  Precision:      {self.metrics['precision']:.4f}")
        print(f"  Recall:         {self.metrics['recall']:.4f}")
        print(f"  F1-Score:       {self.metrics['f1']:.4f}")
        
        print("\n🏥 MEDICAL IMAGING METRICS (CRITICAL):")
        print(f"  Sensitivity:    {self.metrics['sensitivity']:.4f}  ← Detect ulcers")
        print(f"  Specificity:    {self.metrics['specificity']:.4f}  ← Identify healthy")
        print(f"  PPV:            {self.metrics['ppv']:.4f}  ← Precision for positive")
        print(f"  NPV:            {self.metrics['npv']:.4f}  ← Precision for negative")
        print(f"  Balanced Acc:   {self.metrics['balanced_accuracy']:.4f}  ← (Sens+Spec)/2")
        
        print("\n📊 CONFUSION MATRIX:")
        cm = self.metrics['confusion_matrix']
        print(f"  TN: {self.metrics['tn']:4d}  FP: {self.metrics['fp']:4d}")
        print(f"  FN: {self.metrics['fn']:4d}  TP: {self.metrics['tp']:4d}")
        
        print("\n📈 CURVE METRICS:")
        if self.metrics['auc_roc'] is not None:
            print(f"  ROC-AUC:        {self.metrics['auc_roc']:.4f}")
            print(f"  PR-AUC:         {self.metrics['auc_pr']:.4f}")
        else:
            print(f"  ROC-AUC:        N/A (need probabilities)")
            print(f"  PR-AUC:         N/A (need probabilities)")
        
        print("\n🔄 AGREEMENT METRICS:")
        print(f"  MCC:            {self.metrics['mcc']:.4f}  ← Good for imbalanced")
        print(f"  Kappa Score:    {self.metrics['kappa']:.4f}  ← Inter-rater agreement")
        
        print("\n🚨 ERROR RATES:")
        print(f"  FPR:            {self.metrics['fpr']:.4f}  ← Type I error")
        print(f"  FNR:            {self.metrics['fnr']:.4f}  ← Type II error (CRITICAL)")
        
        print("\n📋 CLASSIFICATION REPORT:")
        print(self.metrics['classification_report'])


# =====================
# VISUALIZATION FUNCTIONS
# =====================

def plot_confusion_matrix(y_true, y_pred, model_name, output_dir):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap='Blues', interpolation='nearest')
    
    plt.colorbar(im)
    tick_marks = np.arange(2)
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(['Healthy', 'Ulcer'])
    ax.set_yticklabels(['Healthy', 'Ulcer'])
    
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j],
                   ha="center", va="center",
                   color="white" if cm[i, j] > cm.max() / 2 else "black",
                   fontsize=14, fontweight='bold')
    
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    ax.set_title(f'Confusion Matrix: {model_name}')
    
    plt.tight_layout()
    save_path = output_dir / f"confusion_matrix_{model_name}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return save_path


def plot_roc_curve(y_true, y_probs, model_name, output_dir):
    """Plot and save ROC curve"""
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC={roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve: {model_name}')
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    save_path = output_dir / f"roc_curve_{model_name}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return save_path


def plot_precision_recall_curve(y_true, y_probs, model_name, output_dir):
    """Plot and save Precision-Recall curve"""
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    pr_auc = auc(recall, precision)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, color='green', lw=2, label=f'PR (AUC={pr_auc:.4f})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f'Precision-Recall Curve: {model_name}')
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    
    plt.tight_layout()
    save_path = output_dir / f"pr_curve_{model_name}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return save_path


# =====================
# EVALUATION FUNCTIONS
# =====================

def evaluate_rgb_only(model, test_loader, device):
    """Evaluate RGB-only model"""
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for rgb_batch, labels in tqdm(test_loader, desc="RGB Evaluation"):
            rgb_batch = rgb_batch.to(device)
            output = model(rgb_batch)
            probs = torch.softmax(output, dim=1)[:, 1].cpu().numpy()  # Probability of class 1 (ulcer)
            preds = torch.argmax(output, dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())
    
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def evaluate_thermal_only(model, test_loader, device):
    """Evaluate Thermal-only model"""
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for thermal_batch, labels in tqdm(test_loader, desc="Thermal Evaluation"):
            thermal_batch = thermal_batch.to(device)
            output = model(thermal_batch)
            probs = torch.softmax(output, dim=1)[:, 1].cpu().numpy()  # Probability of class 1 (ulcer)
            preds = torch.argmax(output, dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())
    
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def evaluate_multimodal(model, test_loader, device):
    """Evaluate Multimodal model"""
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for rgb_batch, thermal_batch, labels in tqdm(test_loader, desc="Multimodal Evaluation"):
            rgb_batch = rgb_batch.to(device)
            thermal_batch = thermal_batch.to(device)
            output = model(rgb_batch, thermal_batch)
            probs = torch.softmax(output, dim=1)[:, 1].cpu().numpy()  # Probability of class 1 (ulcer)
            preds = torch.argmax(output, dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())
    
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


# =====================
# MAIN
# =====================

def main():
    print("="*70)
    print("EXTENDED MEDICAL METRICS EVALUATION")
    print("="*70)
    print(f"Device: {DEVICE}\n")
    
    # Load datasets
    print("Loading test datasets...")
    
    # RGB-only dataset
    test_dataset_rgb = RGBDataset(
        data_dir=RGB_DIR,
        split='test',
        transform=rgb_eval_transform
    )
    test_loader_rgb = DataLoader(test_dataset_rgb, batch_size=8, shuffle=False)
    
    # Thermal-only dataset
    test_dataset_thermal = ThermalDataset(
        data_dir=THERMAL_DIR,
        split='test',
        transform=thermal_eval_transform
    )
    test_loader_thermal = DataLoader(test_dataset_thermal, batch_size=8, shuffle=False)
    
    # Multimodal paired dataset
    test_dataset_multimodal = MultimodalDataset(
        rgb_dir=RGB_DIR,
        thermal_dir=THERMAL_DIR,
        split='test',
        transform_rgb=rgb_eval_transform,
        transform_thermal=thermal_eval_transform
    )
    test_loader_multimodal = DataLoader(test_dataset_multimodal, batch_size=8, shuffle=False)
    
    print()
    
    # Store all results for comparison
    all_results = {}
    
    # =====================
    # RGB-ONLY EVALUATION
    # =====================
    print("\n" + "🔴 "*35)
    print("EVALUATING RGB-ONLY MODEL")
    print("🔴 "*35)
    
    rgb_checkpoint = CHECKPOINT_DIR / "checkpoints_rgb_only" / "best_model.pt"
    if rgb_checkpoint.exists():
        print(f"Loading: {rgb_checkpoint}")
        rgb_model = RGBOnlyModel().to(DEVICE)
        checkpoint = torch.load(rgb_checkpoint, map_location=DEVICE, weights_only=False)
        state_dict = checkpoint['model_state_dict']
        
        # Fix naming: backbone -> resnet
        fixed_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('backbone.'):
                new_key = 'resnet.' + key[9:]
                fixed_state_dict[new_key] = value
            else:
                fixed_state_dict[key] = value
        
        rgb_model.load_state_dict(fixed_state_dict, strict=False)
        print("✅ Model loaded successfully!\\n")
        
        y_true, y_pred, y_probs = evaluate_rgb_only(rgb_model, test_loader_rgb, DEVICE)
        
        calculator = MedicalMetricsCalculator(y_true, y_pred, y_probs)
        calculator.print_report("RGB-Only (ResNet50)")
        
        rgb_output = OUTPUT_DIR / "rgb_only"
        rgb_output.mkdir(parents=True, exist_ok=True)
        
        plot_confusion_matrix(y_true, y_pred, "RGB-Only", rgb_output)
        plot_roc_curve(y_true, y_probs, "RGB-Only", rgb_output)
        plot_precision_recall_curve(y_true, y_probs, "RGB-Only", rgb_output)
        
        all_results['RGB-Only'] = calculator.get_all_metrics()
        
        torch.save({
            'y_true': y_true,
            'y_pred': y_pred,
            'y_probs': y_probs,
            'metrics': calculator.get_all_metrics()
        }, rgb_output / "results.pt")
        print("\n✓ Results saved to rgb_only/results.pt")
    else:
        print(f"❌ Checkpoint not found: {rgb_checkpoint}")
    
    # =====================
    # THERMAL-ONLY EVALUATION
    # =====================
    print("\n" + "🔵 "*35)
    print("EVALUATING THERMAL-ONLY MODEL")
    print("🔵 "*35)
    
    thermal_checkpoint = CHECKPOINT_DIR / "checkpoints_thermal_only" / "best_model.pt"
    if thermal_checkpoint.exists():
        print(f"Loading: {thermal_checkpoint}")
        thermal_model = ThermalOnlyModel().to(DEVICE)
        checkpoint = torch.load(thermal_checkpoint, map_location=DEVICE, weights_only=False)
        state_dict = checkpoint['model_state_dict']
        
        # Fix naming: backbone -> vit
        fixed_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('backbone.'):
                new_key = 'vit.' + key[9:]
                fixed_state_dict[new_key] = value
            else:
                fixed_state_dict[key] = value
        
        thermal_model.load_state_dict(fixed_state_dict, strict=False)
        print("✅ Model loaded successfully!\\n")
        
        y_true, y_pred, y_probs = evaluate_thermal_only(thermal_model, test_loader_thermal, DEVICE)
        
        calculator = MedicalMetricsCalculator(y_true, y_pred, y_probs)
        calculator.print_report("Thermal-Only (ViT)")
        
        thermal_output = OUTPUT_DIR / "thermal_only"
        thermal_output.mkdir(parents=True, exist_ok=True)
        
        plot_confusion_matrix(y_true, y_pred, "Thermal-Only", thermal_output)
        plot_roc_curve(y_true, y_probs, "Thermal-Only", thermal_output)
        plot_precision_recall_curve(y_true, y_probs, "Thermal-Only", thermal_output)
        
        all_results['Thermal-Only'] = calculator.get_all_metrics()
        
        torch.save({
            'y_true': y_true,
            'y_pred': y_pred,
            'y_probs': y_probs,
            'metrics': calculator.get_all_metrics()
        }, thermal_output / "results.pt")
        print("\n✓ Results saved to thermal_only/results.pt")
    else:
        print(f"❌ Checkpoint not found: {thermal_checkpoint}")
    
    # =====================
    # MULTIMODAL EVALUATION
    # =====================
    print("\n" + "🟣 "*35)
    print("EVALUATING MULTIMODAL FUSION MODEL")
    print("🟣 "*35)
    
    multimodal_checkpoint = CHECKPOINT_DIR / "checkpoints_multimodal" / "best_model.pt"
    if multimodal_checkpoint.exists():
        print(f"Loading: {multimodal_checkpoint}")
        multimodal_model = MultimodalFusionModel().to(DEVICE)
        checkpoint = torch.load(multimodal_checkpoint, map_location=DEVICE)
        state_dict = checkpoint['model_state_dict']
        
        # Fix naming: for multimodal, RGB branch uses backbone -> resnet, Thermal branch keeps vit
        fixed_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('backbone.'):
                # RGB backbone -> resnet
                new_key = 'resnet.' + key[9:]
                fixed_state_dict[new_key] = value
            elif key.startswith('vit_backbone.'):
                # Thermal backbone -> vit
                new_key = 'vit.' + key[13:]
                fixed_state_dict[new_key] = value
            else:
                fixed_state_dict[key] = value
        
        multimodal_model.load_state_dict(fixed_state_dict, strict=False)
        print("✅ Model loaded successfully!\n")
        
        # Use multimodal-style paired test loader (matches training pairing)
        y_true, y_pred, y_probs = evaluate_multimodal(multimodal_model, test_loader_multimodal, DEVICE)
        
        calculator = MedicalMetricsCalculator(y_true, y_pred, y_probs)
        calculator.print_report("Multimodal Fusion (ResNet50 + ViT)")
        
        mm_output = OUTPUT_DIR / "multimodal"
        mm_output.mkdir(parents=True, exist_ok=True)
        
        plot_confusion_matrix(y_true, y_pred, "Multimodal", mm_output)
        plot_roc_curve(y_true, y_probs, "Multimodal", mm_output)
        plot_precision_recall_curve(y_true, y_probs, "Multimodal", mm_output)
        
        all_results['Multimodal'] = calculator.get_all_metrics()
        
        torch.save({
            'y_true': y_true,
            'y_pred': y_pred,
            'y_probs': y_probs,
            'metrics': calculator.get_all_metrics()
        }, mm_output / "results.pt")
        print("\n✓ Results saved to multimodal/results.pt")
    else:
        print(f"❌ Checkpoint not found: {multimodal_checkpoint}")
    
    # =====================
    # SUMMARY COMPARISON
    # =====================
    print("\n" + "="*70)
    print("SUMMARY COMPARISON")
    print("="*70)
    
    if all_results:
        print("\nF1-Scores:")
        for model_name, metrics in all_results.items():
            print(f"  {model_name:20s}: {metrics['f1']:.4f}")
        
        print("\nSensitivity (Detect Ulcers):")
        for model_name, metrics in all_results.items():
            print(f"  {model_name:20s}: {metrics['sensitivity']:.4f}")
        
        print("\nSpecificity (Identify Healthy):")
        for model_name, metrics in all_results.items():
            print(f"  {model_name:20s}: {metrics['specificity']:.4f}")
    
    print("\n" + "="*70)
    print(f"✅ METRICS SAVED TO: {OUTPUT_DIR}")
    print("="*70)


if __name__ == "__main__":
    main()