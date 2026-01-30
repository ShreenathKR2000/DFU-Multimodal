#!/usr/bin/env python3
"""
TEST-TIME AUGMENTATION (TTA) EVALUATION
========================================

Evaluates model robustness by applying moderate augmentations at test time.
Compares clean accuracy vs TTA accuracy to measure generalization.

NOW SUPPORTS:
✅ RGB-only (ResNet50)
✅ Thermal-only (ViT)
✅ Multimodal Fusion (ResNet50 + ViT)
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, 
    classification_report, roc_auc_score
)
from tqdm import tqdm
import timm

# Project paths
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))
from scripts.dataloader import DFUPairedDataset
from PIL import Image
import torchvision.transforms.functional as TF


def load_checkpoint_flexible(model, checkpoint_path, device='cuda'):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)

    if not isinstance(state_dict, dict):
        return False

    model_state = model.state_dict()

    fixed_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('backbone.', 'resnet.')
        fixed_state_dict[new_key] = value

    # Load matching keys while skipping size-mismatched heads
    for key, value in fixed_state_dict.items():
        if key not in model_state:
            continue
        try:
            if value.shape != model_state[key].shape:
                if 'fc' in key or 'classifier' in key:
                    continue
                else:
                    continue
        except Exception:
            # If shapes not available, skip
            continue
        model_state[key] = value

    model.load_state_dict(model_state, strict=False)
    return True


# =====================
# CONFIGURATION
# =====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 6
CHECKPOINT_DIR = project_root / "logs"
DATA_DIR = Path.home() / "CompVis" / "Dataset" / "data"


# =====================
# MODEL DEFINITIONS (Same as training)
# =====================

class RGBOnlyModel(nn.Module):
    """ResNet50 for RGB-only classification"""
    def __init__(self):
        super().__init__()
        self.resnet = torch.hub.load('pytorch/vision:v0.13.1', 'resnet50', pretrained=True)
        self.resnet.fc = nn.Linear(2048, 1)
    
    def forward(self, x):
        return self.resnet(x)


class ThermalOnlyModel(nn.Module):
    """ViT for Thermal-only classification"""
    def __init__(self):
        super().__init__()
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
        self.classifier = nn.Linear(768, 1)
    
    def forward(self, x):
        features = self.vit(x)
        return self.classifier(features)


class MLPFusion(nn.Module):
    """Late fusion classifier"""
    def __init__(self, rgb_feat_dim=2048, thermal_feat_dim=768, hidden_dim=512):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(rgb_feat_dim + thermal_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, rgb_feat, thermal_feat):
        fused = torch.cat([rgb_feat, thermal_feat], dim=1)
        return self.classifier(fused)


class MultimodalFusionModel(nn.Module):
    """Complete multimodal model"""
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
# AUGMENTATIONS FOR TTA
# =====================

def get_light_augmentation_transforms():
    """Light augmentation transforms for TTA"""
    rgb_tta = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    thermal_tta = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    
    return rgb_tta, thermal_tta


def get_clean_transforms():
    """Clean transforms without augmentation"""
    rgb_clean = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    thermal_clean = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    
    return rgb_clean, thermal_clean


# =====================
# TTA EVALUATION FUNCTIONS
# =====================

def evaluate_rgb_only_with_tta(model, test_loader, device, num_tta=5, use_augmentation=True):
    """Evaluate RGB-only model with TTA"""
    model.eval()
    rgb_tta, _ = get_light_augmentation_transforms()
    rgb_clean, _ = get_clean_transforms()
    
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for rgb_batch, labels in tqdm(test_loader, desc="RGB-only TTA evaluation"):
            batch_size = len(labels)
            batch_preds = []
            batch_probs = []
            
            for img_idx in range(batch_size):
                single_rgb = rgb_batch[img_idx]  # PIL image
                sample_preds = []
                sample_probs = []
                
                for aug_idx in range(num_tta):
                    if use_augmentation:
                        rgb_tensor = rgb_tta(single_rgb).unsqueeze(0).to(device)
                    else:
                        rgb_tensor = rgb_clean(single_rgb).unsqueeze(0).to(device)
                    
                    output = model(rgb_tensor)
                    prob = torch.sigmoid(output).cpu().numpy()[0, 0]
                    pred = (prob > 0.5).astype(int)
                    
                    sample_preds.append(pred)
                    sample_probs.append(prob)
                
                avg_pred = np.mean(sample_preds)
                avg_prob = np.mean(sample_probs)
                
                batch_preds.append((avg_pred > 0.5).astype(int))
                batch_probs.append(avg_prob)
            
            all_preds.extend(batch_preds)
            all_probs.extend(batch_probs)
            all_labels.extend(labels.numpy())
    
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    cm = confusion_matrix(all_labels, all_preds)
    
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'auc': auc,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'probabilities': all_probs,
        'labels': all_labels
    }


def evaluate_thermal_only_with_tta(model, test_loader, device, num_tta=5, use_augmentation=True):
    """Evaluate Thermal-only model with TTA"""
    model.eval()
    _, thermal_tta = get_light_augmentation_transforms()
    _, thermal_clean = get_clean_transforms()
    
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for thermal_batch, labels in tqdm(test_loader, desc="Thermal-only TTA evaluation"):
            batch_size = len(labels)
            batch_preds = []
            batch_probs = []
            
            for img_idx in range(batch_size):
                single_thermal = thermal_batch[img_idx]
                sample_preds = []
                sample_probs = []
                
                for aug_idx in range(num_tta):
                    if use_augmentation:
                        thermal_tensor = thermal_tta(single_thermal).unsqueeze(0).to(device)
                    else:
                        thermal_tensor = thermal_clean(single_thermal).unsqueeze(0).to(device)
                    
                    output = model(thermal_tensor)
                    prob = torch.sigmoid(output).cpu().numpy()[0, 0]
                    pred = (prob > 0.5).astype(int)
                    
                    sample_preds.append(pred)
                    sample_probs.append(prob)
                
                avg_pred = np.mean(sample_preds)
                avg_prob = np.mean(sample_probs)
                
                batch_preds.append((avg_pred > 0.5).astype(int))
                batch_probs.append(avg_prob)
            
            all_preds.extend(batch_preds)
            all_probs.extend(batch_probs)
            all_labels.extend(labels.numpy())
    
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    cm = confusion_matrix(all_labels, all_preds)
    
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'auc': auc,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'probabilities': all_probs,
        'labels': all_labels
    }


def evaluate_multimodal_with_tta(model, test_loader, device, num_tta=5, use_augmentation=True):
    """Evaluate Multimodal Fusion model with TTA"""
    model.eval()
    rgb_tta, thermal_tta = get_light_augmentation_transforms()
    rgb_clean, thermal_clean = get_clean_transforms()
    
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for rgb_batch, thermal_batch, labels in tqdm(test_loader, desc="Multimodal TTA evaluation"):
            batch_size = len(labels)
            batch_preds = []
            batch_probs = []
            
            for img_idx in range(batch_size):
                single_rgb = rgb_batch[img_idx]
                single_thermal = thermal_batch[img_idx]
                sample_preds = []
                sample_probs = []
                
                for aug_idx in range(num_tta):
                    if use_augmentation:
                        rgb_tensor = rgb_tta(single_rgb).unsqueeze(0).to(device)
                        thermal_tensor = thermal_tta(single_thermal).unsqueeze(0).to(device)
                    else:
                        rgb_tensor = rgb_clean(single_rgb).unsqueeze(0).to(device)
                        thermal_tensor = thermal_clean(single_thermal).unsqueeze(0).to(device)
                    
                    output = model(rgb_tensor, thermal_tensor)
                    prob = torch.sigmoid(output).cpu().numpy()[0, 0]
                    pred = (prob > 0.5).astype(int)
                    
                    sample_preds.append(pred)
                    sample_probs.append(prob)
                
                avg_pred = np.mean(sample_preds)
                avg_prob = np.mean(sample_probs)
                
                batch_preds.append((avg_pred > 0.5).astype(int))
                batch_probs.append(avg_prob)
            
            all_preds.extend(batch_preds)
            all_probs.extend(batch_probs)
            all_labels.extend(labels.numpy())
    
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    cm = confusion_matrix(all_labels, all_preds)
    
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'auc': auc,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'probabilities': all_probs,
        'labels': all_labels
    }


def print_tta_comparison(clean_metrics, tta_metrics, model_name):
    """Print comparison of clean vs TTA results"""
    print("\n" + "="*70)
    print(f"TEST-TIME AUGMENTATION EVALUATION: {model_name}")
    print("="*70)
    
    print("\n📊 CLEAN EVALUATION (No Augmentation):")
    print(f"  Accuracy:    {clean_metrics['accuracy']:.4f}")
    print(f"  F1-Score:    {clean_metrics['f1']:.4f}")
    print(f"  AUC-ROC:     {clean_metrics['auc']:.4f}")
    print(f"  Sensitivity: {clean_metrics['sensitivity']:.4f}")
    print(f"  Specificity: {clean_metrics['specificity']:.4f}")
    
    print("\n🔄 TTA EVALUATION (5× Augmented):")
    print(f"  Accuracy:    {tta_metrics['accuracy']:.4f}")
    print(f"  F1-Score:    {tta_metrics['f1']:.4f}")
    print(f"  AUC-ROC:     {tta_metrics['auc']:.4f}")
    print(f"  Sensitivity: {tta_metrics['sensitivity']:.4f}")
    print(f"  Specificity: {tta_metrics['specificity']:.4f}")
    
    print("\n📈 ROBUSTNESS COMPARISON:")
    acc_drop = clean_metrics['accuracy'] - tta_metrics['accuracy']
    f1_drop = clean_metrics['f1'] - tta_metrics['f1']
    print(f"  Accuracy drop:  {acc_drop:.4f} ({acc_drop*100:.2f}%)")
    print(f"  F1-Score drop:  {f1_drop:.4f}")
    
    if abs(acc_drop) < 0.05:
        print(f"\n  ✅ ROBUST: Model generalizes well to variations")
    elif abs(acc_drop) < 0.15:
        print(f"\n  ⚠️  MODERATE: Some performance drop with augmentation")
    else:
        print(f"\n  ❌ NOT ROBUST: Large performance drop suggests overfitting")
    
    print("\n🎯 CONFUSION MATRICES:")
    print("\nClean:")
    print(clean_metrics['confusion_matrix'])
    print("\nTTA:")
    print(tta_metrics['confusion_matrix'])


# =====================
# MAIN
# =====================

def main():
    print("="*70)
    print("TEST-TIME AUGMENTATION ROBUSTNESS EVALUATION")
    print("="*70)
    print(f"Device: {DEVICE}\n")
    
    # Load test dataset
    print("Loading test dataset...")
    # The project dataset provides a paired dataset. Create simple collate
    # functions so DataLoader returns lists of PIL images (so our TTA
    # transforms can be applied per-sample) and a tensor of labels.
    def collate_single(batch):
        imgs = [item[0] for item in batch]
        labels = torch.as_tensor([int(item[1]) for item in batch], dtype=torch.long)
        return imgs, labels

    def collate_paired(batch):
        rgbs = [item[0] for item in batch]
        thermals = [item[1] for item in batch]
        labels = torch.as_tensor([int(item[2]) for item in batch], dtype=torch.long)
        return rgbs, thermals, labels

    # Build datasets/loaders using the paired dataset (no transforms applied)
    # We'll apply transforms inside the evaluation functions per-sample.
    test_dataset_rgb = DFUPairedDataset(
        data_dir=DATA_DIR,
        split='test',
        transform_rgb=None,
        transform_thermal=None
    )
    # Wrap the paired dataset in a simple view that returns only the rgb and label
    class SingleRGBView(torch.utils.data.Dataset):
        def __init__(self, paired):
            self.paired = paired
        def __len__(self):
            return len(self.paired)
        def __getitem__(self, idx):
            rgb, _, label = self.paired[idx]
            return rgb, label

    class SingleThermalView(torch.utils.data.Dataset):
        def __init__(self, paired):
            self.paired = paired
        def __len__(self):
            return len(self.paired)
        def __getitem__(self, idx):
            _, thermal, label = self.paired[idx]
            return thermal, label

    test_loader_rgb = DataLoader(SingleRGBView(test_dataset_rgb), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_single)
    test_loader_thermal = DataLoader(SingleThermalView(test_dataset_rgb), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_single)
    test_loader_paired = DataLoader(test_dataset_rgb, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_paired)
    
    print(f"✓ Test set size: {len(test_dataset_rgb)} images\n")
    
    # =====================
    # EVALUATE RGB-ONLY
    # =====================
    print("\n" + "🔴 "*35)
    print("EVALUATING RGB-ONLY MODEL (ResNet50)")
    print("🔴 "*35)
    
    rgb_checkpoint_path = CHECKPOINT_DIR / "checkpoints_rgb_only" / "best_model.pt"
    if rgb_checkpoint_path.exists():
        print(f"Loading checkpoint: {rgb_checkpoint_path}")
        rgb_model = RGBOnlyModel().to(DEVICE)
        # Load checkpoint flexibly to tolerate naming/shape differences
        if not load_checkpoint_flexible(rgb_model, rgb_checkpoint_path, DEVICE):
            checkpoint = torch.load(rgb_checkpoint_path, map_location=DEVICE)
            rgb_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        rgb_model.eval()
        
        # Clean evaluation
        print("\nRunning clean evaluation...")
        clean_metrics_rgb = evaluate_rgb_only_with_tta(
            rgb_model, test_loader_rgb, DEVICE, num_tta=1, use_augmentation=False
        )
        
        # TTA evaluation
        print("\nRunning TTA evaluation...")
        tta_metrics_rgb = evaluate_rgb_only_with_tta(
            rgb_model, test_loader_rgb, DEVICE, num_tta=5, use_augmentation=True
        )
        
        print_tta_comparison(clean_metrics_rgb, tta_metrics_rgb, "RGB-Only (ResNet50)")
        
        # Save results
        torch.save({
            'model': 'RGB-Only',
            'clean_metrics': clean_metrics_rgb,
            'tta_metrics': tta_metrics_rgb
        }, CHECKPOINT_DIR / "checkpoints_rgb_only" / "tta_results.pt")
        print("\n✓ Results saved to tta_results.pt")
    else:
        print(f"❌ Checkpoint not found: {rgb_checkpoint_path}")
    
    # =====================
    # EVALUATE THERMAL-ONLY
    # =====================
    print("\n\n" + "🔵 "*35)
    print("EVALUATING THERMAL-ONLY MODEL (ViT)")
    print("🔵 "*35)
    
    thermal_checkpoint_path = CHECKPOINT_DIR / "checkpoints_thermal_only" / "best_model.pt"
    if thermal_checkpoint_path.exists():
        print(f"Loading checkpoint: {thermal_checkpoint_path}")
        thermal_model = ThermalOnlyModel().to(DEVICE)
        if not load_checkpoint_flexible(thermal_model, thermal_checkpoint_path, DEVICE):
            checkpoint = torch.load(thermal_checkpoint_path, map_location=DEVICE)
            thermal_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        thermal_model.eval()
        
        # Clean evaluation
        print("\nRunning clean evaluation...")
        clean_metrics_thermal = evaluate_thermal_only_with_tta(
            thermal_model, test_loader_thermal, DEVICE, num_tta=1, use_augmentation=False
        )
        
        # TTA evaluation
        print("\nRunning TTA evaluation...")
        tta_metrics_thermal = evaluate_thermal_only_with_tta(
            thermal_model, test_loader_thermal, DEVICE, num_tta=5, use_augmentation=True
        )
        
        print_tta_comparison(clean_metrics_thermal, tta_metrics_thermal, "Thermal-Only (ViT)")
        
        # Save results
        torch.save({
            'model': 'Thermal-Only',
            'clean_metrics': clean_metrics_thermal,
            'tta_metrics': tta_metrics_thermal
        }, CHECKPOINT_DIR / "checkpoints_thermal_only" / "tta_results.pt")
        print("\n✓ Results saved to tta_results.pt")
    else:
        print(f"❌ Checkpoint not found: {thermal_checkpoint_path}")
    
    # =====================
    # EVALUATE MULTIMODAL
    # =====================
    print("\n\n" + "🟣 "*35)
    print("EVALUATING MULTIMODAL FUSION MODEL (ResNet50 + ViT)")
    print("🟣 "*35)
    
    multimodal_checkpoint_path = CHECKPOINT_DIR / "checkpoints_multimodal" / "best_model.pt"
    if multimodal_checkpoint_path.exists():
        print(f"Loading checkpoint: {multimodal_checkpoint_path}")
        multimodal_model = MultimodalFusionModel().to(DEVICE)
        # multimodal checkpoints may include backbone prefixes; try flexible loader
        if not load_checkpoint_flexible(multimodal_model, multimodal_checkpoint_path, DEVICE):
            checkpoint = torch.load(multimodal_checkpoint_path, map_location=DEVICE)
            multimodal_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        multimodal_model.eval()
        
        # Clean evaluation
        print("\nRunning clean evaluation...")
        clean_metrics_mm = evaluate_multimodal_with_tta(
            multimodal_model, test_loader_paired, DEVICE, num_tta=1, use_augmentation=False
        )
        
        # TTA evaluation
        print("\nRunning TTA evaluation...")
        tta_metrics_mm = evaluate_multimodal_with_tta(
            multimodal_model, test_loader_paired, DEVICE, num_tta=5, use_augmentation=True
        )
        
        print_tta_comparison(clean_metrics_mm, tta_metrics_mm, "Multimodal Fusion")
        
        # Save results
        torch.save({
            'model': 'Multimodal Fusion',
            'clean_metrics': clean_metrics_mm,
            'tta_metrics': tta_metrics_mm
        }, CHECKPOINT_DIR / "checkpoints_multimodal" / "tta_results.pt")
        print("\n✓ Results saved to tta_results.pt")
    else:
        print(f"❌ Checkpoint not found: {multimodal_checkpoint_path}")
    
    print("\n" + "="*70)
    print("✅ TTA EVALUATION COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
