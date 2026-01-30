#!/usr/bin/env python3
"""
GRAD-CAM VISUALIZATION FOR DFU MODELS
=====================================

Generates visual explanations showing which image regions the model uses for predictions.

NOW SUPPORTS:
✅ RGB-only (ResNet50)
✅ Thermal-only (ViT)  
✅ Multimodal Fusion (ResNet50 + ViT)
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
from matplotlib import pyplot as plt
from tqdm import tqdm
import timm
from torch.utils.data import DataLoader

# Project paths
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))
from PIL import Image


def load_checkpoint_flexible(model, checkpoint_path, device='cuda'):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)

    if not isinstance(state_dict, dict):
        return False

    # Fix naming: backbone -> resnet or vit
    fixed_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('backbone.'):
            new_key = key[9:]  # Remove 'backbone.'
            if hasattr(model, 'resnet'):
                new_key = 'resnet.' + new_key
            elif hasattr(model, 'vit'):
                new_key = 'vit.' + new_key
            fixed_state_dict[new_key] = value
        else:
            fixed_state_dict[key] = value

    model.load_state_dict(fixed_state_dict, strict=False)
    return True


# =====================
# CONFIGURATION
# =====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_DIR = project_root / "logs"
DATA_DIR = Path.home() / "CompVis" / "Dataset" / "data"
RGB_DIR = DATA_DIR / "rgb"
THERMAL_DIR = DATA_DIR / "thermal"
OUTPUT_DIR = CHECKPOINT_DIR / "grad_cam_visualizations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =====================
# DATASET DEFINITIONS
# =====================

class RGBDataset(torch.utils.data.Dataset):
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


class ThermalDataset(torch.utils.data.Dataset):
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


class MultimodalDataset(torch.utils.data.Dataset):
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
        import random
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
# MODEL DEFINITIONS
# =====================

class RGBOnlyModel(nn.Module):
    """ResNet50 for RGB-only"""
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
    """Late fusion classifier"""
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
    """Complete multimodal model"""
    def __init__(self, num_classes=2):
        super().__init__()
        self.resnet = torch.hub.load('pytorch/vision:v0.13.1', 'resnet50', pretrained=True)
        self.resnet.fc = nn.Identity()
        
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
        
        self.fusion = MLPFusion(rgb_feat_dim=2048, thermal_feat_dim=768, hidden_dim=512, num_classes=num_classes)
    
    def forward(self, rgb, thermal):
        rgb_feat = self.resnet(rgb)
        thermal_feat = self.vit(thermal)
        output = self.fusion(rgb_feat, thermal_feat)
        return output


# =====================
# GRAD-CAM IMPLEMENTATION
# =====================

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping
    """
    
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers if isinstance(target_layers, list) else [target_layers]
        self.activations = {}
        self.gradients = {}
        self._register_hooks()
    
    def _register_hooks(self):
        """Register hooks to capture activations and gradients"""
        def get_activation(name):
            def hook(module, input, output):
                # Store activation (not detached) and register a hook to capture its gradient
                self.activations[name] = output
                try:
                    def save_grad(grad):
                        self.gradients[name] = grad
                    output.register_hook(save_grad)
                except Exception:
                    pass
            return hook

        # Register forward hooks for all target layers; gradients are captured via
        # hooks attached to the activation tensors during the forward pass.
        for name, module in self.model.named_modules():
            if any(layer in name for layer in self.target_layers):
                module.register_forward_hook(get_activation(name))
    
    def generate_cam(self, input_tensor, class_idx=None):
        """
        Generate Grad-CAM heatmap
        
        Args:
            input_tensor: Input (1, C, H, W)
            class_idx: Target class
        
        Returns:
            cam: Heatmap (H, W)
        """
        self.model.eval()
        
        # Ensure gradient tracking is enabled for CAM generation
        # Work on a copy of the input so we can get input gradients if needed
        input_for_grad = input_tensor.clone().detach().requires_grad_(True)
        with torch.enable_grad():
            output = self.model(input_for_grad)

            if class_idx is None:
                class_idx = (output > 0.5).long().item()

            # Backward pass
            self.model.zero_grad()
            if input_for_grad.grad is not None:
                input_for_grad.grad.zero_()
            score = output[0, 0]  # For binary classification
            score.backward()
        
        # Get last target layer
        layer_name = None
        for name, _ in self.model.named_modules():
            if any(layer in name for layer in self.target_layers):
                layer_name = name
        
        if layer_name is None:
            print(f"Warning: Could not find layer {self.target_layers}")
            return None
        
        activations = self.activations[layer_name]
        gradients = self.gradients.get(layer_name, None)

        # If activations are not 4D (e.g., ViT blocks: (B, N, C)), fall back
        # to input-gradient saliency which works for any model.
        if activations.ndim != 4 or gradients is None or gradients.ndim != 4:
            # Use input gradients as saliency map
            if input_for_grad.grad is None:
                cam = torch.zeros(input_tensor.shape[2:], device=input_tensor.device)
            else:
                saliency = input_for_grad.grad.detach().abs()  # (1, C, H, W)
                cam = saliency.mean(dim=1)[0]
                if cam.max() > 0:
                    cam = cam / cam.max()
                cam = cam.cpu().numpy()
                return cam

        # Calculate weights (average gradient over spatial dimensions)
        weights = gradients.mean(dim=(2, 3))  # (1, C)

        # Generate CAM (handle possible channel mismatches)
        cam = torch.zeros(activations.shape[2:], device=activations.device)
        n_channels = min(activations.shape[1], weights.shape[1])
        for i in range(n_channels):
            cam += weights[0, i] * activations[0, i, :, :]
        
        # ReLU to get positive activations
        cam = F.relu(cam)
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam.cpu().numpy()


def overlay_gradcam_on_image(image_np, cam, alpha=0.5):
    """
    Overlay Grad-CAM on image
    
    Args:
        image_np: Original image (H, W, 3) normalized [0, 1] or [0, 255]
        cam: Grad-CAM heatmap (H, W) normalized [0, 1]
        alpha: Transparency
    
    Returns:
        overlay: Overlaid image (H, W, 3)
    """
    # Ensure image is uint8
    if image_np.dtype != np.uint8:
        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)
        else:
            image_np = image_np.astype(np.uint8)
    
    # Resize CAM to match image
    cam_resized = cv2.resize(cam, (image_np.shape[1], image_np.shape[0]))
    cam_resized = (cam_resized * 255).astype(np.uint8)
    
    # Apply colormap
    heatmap = cv2.applyColorMap(cam_resized, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Overlay
    overlay = cv2.addWeighted(image_np, 1 - alpha, heatmap, alpha, 0)
    
    return overlay, heatmap


def visualize_rgb_only(model, rgb_image, device, label):
    """Visualize Grad-CAM for RGB-only model"""
    
    rgb_image = rgb_image.to(device)
    
    # Get prediction
    with torch.no_grad():
        output = model(rgb_image)
        probs = torch.softmax(output, dim=1)
        prediction = torch.argmax(probs, dim=1).item()
        confidence = probs[0, prediction].item()
    
    # Generate Grad-CAM
    grad_cam = GradCAM(model, target_layers=['layer4'])
    cam = grad_cam.generate_cam(rgb_image, class_idx=prediction)
    
    if cam is None:
        return None
    
    # Denormalize image
    img_np = rgb_image[0].permute(1, 2, 0).cpu().numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
    img_np = (img_np * 255).astype(np.uint8)
    
    # Create overlay
    overlay, heatmap = overlay_gradcam_on_image(img_np, cam)
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img_np)
    axes[0].set_title("Original RGB Image")
    axes[0].axis('off')
    
    axes[1].imshow(heatmap)
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis('off')
    
    axes[2].imshow(overlay)
    axes[2].set_title(f"Overlay\nPred: {'Ulcer' if prediction == 1 else 'Healthy'} ({confidence:.3f})")
    axes[2].axis('off')
    
    plt.suptitle("RGB-Only Model Grad-CAM", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


def visualize_thermal_only(model, thermal_image, device, label):
    """Visualize Grad-CAM for Thermal-only model"""
    
    thermal_image = thermal_image.to(device)
    
    # Get prediction
    with torch.no_grad():
        output = model(thermal_image)
        probs = torch.softmax(output, dim=1)
        prediction = torch.argmax(probs, dim=1).item()
        confidence = probs[0, prediction].item()
    
    # Generate Grad-CAM (for ViT, use blocks)
    grad_cam = GradCAM(model.vit, target_layers=['blocks'])
    cam = grad_cam.generate_cam(thermal_image, class_idx=prediction)
    
    if cam is None:
        return None
    
    # Denormalize image
    img_np = thermal_image[0].permute(1, 2, 0).cpu().numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
    img_np = (img_np * 255).astype(np.uint8)
    
    # Create overlay
    overlay, heatmap = overlay_gradcam_on_image(img_np, cam)
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img_np, cmap='gray')
    axes[0].set_title("Original Thermal Image")
    axes[0].axis('off')
    
    axes[1].imshow(heatmap)
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis('off')
    
    axes[2].imshow(overlay)
    axes[2].set_title(f"Overlay\nPred: {'Ulcer' if prediction == 1 else 'Healthy'} ({confidence:.3f})")
    axes[2].axis('off')
    
    plt.suptitle("Thermal-Only Model Grad-CAM", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


def visualize_multimodal(model, rgb_image, thermal_image, device, label):
    """Visualize Grad-CAM for both branches in Multimodal model"""
    
    rgb_image = rgb_image.to(device)
    thermal_image = thermal_image.to(device)
    
    # Get prediction
    with torch.no_grad():
        output = model(rgb_image, thermal_image)
        probs = torch.softmax(output, dim=1)
        prediction = torch.argmax(probs, dim=1).item()
        confidence = probs[0, prediction].item()
    
    # RGB Grad-CAM
    rgb_grad_cam = GradCAM(model.resnet, target_layers=['layer4'])
    rgb_cam = rgb_grad_cam.generate_cam(rgb_image, class_idx=prediction)
    
    # Thermal Grad-CAM
    thermal_grad_cam = GradCAM(model.vit, target_layers=['blocks'])
    thermal_cam = thermal_grad_cam.generate_cam(thermal_image, class_idx=prediction)
    
    if rgb_cam is None or thermal_cam is None:
        return None
    
    # Denormalize images
    rgb_np = rgb_image[0].permute(1, 2, 0).cpu().numpy()
    rgb_np = (rgb_np - rgb_np.min()) / (rgb_np.max() - rgb_np.min())
    rgb_np = (rgb_np * 255).astype(np.uint8)
    
    thermal_np = thermal_image[0].permute(1, 2, 0).cpu().numpy()
    thermal_np = (thermal_np - thermal_np.min()) / (thermal_np.max() - thermal_np.min())
    thermal_np = (thermal_np * 255).astype(np.uint8)
    
    # Create overlays
    rgb_overlay, rgb_heatmap = overlay_gradcam_on_image(rgb_np, rgb_cam)
    thermal_overlay, thermal_heatmap = overlay_gradcam_on_image(thermal_np, thermal_cam)
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # RGB row
    axes[0, 0].imshow(rgb_np)
    axes[0, 0].set_title("RGB Image")
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(rgb_heatmap)
    axes[0, 1].set_title("RGB Grad-CAM")
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(rgb_overlay)
    axes[0, 2].set_title("RGB Overlay")
    axes[0, 2].axis('off')
    
    # Thermal row
    axes[1, 0].imshow(thermal_np, cmap='gray')
    axes[1, 0].set_title("Thermal Image")
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(thermal_heatmap)
    axes[1, 1].set_title("Thermal Grad-CAM")
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(thermal_overlay)
    axes[1, 2].set_title("Thermal Overlay")
    axes[1, 2].axis('off')
    
    pred_text = "Ulcer" if prediction == 1 else "Healthy"
    plt.suptitle(f"Multimodal Fusion Grad-CAM\nPrediction: {pred_text} (Confidence: {confidence:.3f})",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


# =====================
# MAIN VISUALIZATION PIPELINE
# =====================

def main():
    print("="*70)
    print("GRAD-CAM VISUALIZATION FOR DFU MODELS")
    print("="*70)
    print(f"Device: {DEVICE}\n")
    
    # Load test datasets
    print("Loading test datasets...")
    
    # Define transforms for visualization (minimal preprocessing)
    from torchvision import transforms
    rgb_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    thermal_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    # RGB-only dataset
    test_dataset_rgb = RGBDataset(
        data_dir=RGB_DIR,
        split='test',
        transform=rgb_transform
    )
    test_loader_rgb = DataLoader(test_dataset_rgb, batch_size=1, shuffle=False)
    
    # Thermal-only dataset
    test_dataset_thermal = ThermalDataset(
        data_dir=THERMAL_DIR,
        split='test',
        transform=thermal_transform
    )
    test_loader_thermal = DataLoader(test_dataset_thermal, batch_size=1, shuffle=False)
    
    # Multimodal paired dataset
    test_dataset_multimodal = MultimodalDataset(
        rgb_dir=RGB_DIR,
        thermal_dir=THERMAL_DIR,
        split='test',
        transform_rgb=rgb_transform,
        transform_thermal=thermal_transform
    )
    test_loader_multimodal = DataLoader(test_dataset_multimodal, batch_size=1, shuffle=False)
    
    # Select 5 healthy + 5 ulcer samples for balanced visualization
    num_healthy = 5
    num_ulcer = 5
    print(f"✓ Will visualize {num_healthy} healthy + {num_ulcer} ulcer samples per model\n")
    
    # =====================
    # RGB-ONLY VISUALIZATION
    # =====================
    print("\n" + "🔴 "*35)
    print("VISUALIZING RGB-ONLY MODEL")
    print("🔴 "*35)
    
    rgb_checkpoint = CHECKPOINT_DIR / "checkpoints_rgb_only" / "best_model.pt"
    if rgb_checkpoint.exists():
        print(f"Loading: {rgb_checkpoint}")
        rgb_model = RGBOnlyModel().to(DEVICE)
        if not load_checkpoint_flexible(rgb_model, rgb_checkpoint, DEVICE):
            checkpoint = torch.load(rgb_checkpoint, map_location=DEVICE)
            rgb_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        rgb_model.eval()
        
        rgb_output_dir = OUTPUT_DIR / "rgb_only"
        rgb_output_dir.mkdir(parents=True, exist_ok=True)
        
        healthy_count = 0
        ulcer_count = 0
        sample_idx = 0
        for rgb_batch, labels in test_loader_rgb:
            label = labels[0].item()
            
            # Skip if we have enough of this class
            if label == 0 and healthy_count >= num_healthy:
                continue
            if label == 1 and ulcer_count >= num_ulcer:
                continue
            if healthy_count >= num_healthy and ulcer_count >= num_ulcer:
                break
            
            with torch.no_grad():
                fig = visualize_rgb_only(rgb_model, rgb_batch, DEVICE, label)
            
            if fig is not None:
                class_name = 'healthy' if label == 0 else 'ulcer'
                class_count = healthy_count if label == 0 else ulcer_count
                save_path = rgb_output_dir / f"{class_name}_{class_count:02d}.png"
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"  ✓ Saved {save_path.name}")
                
                if label == 0:
                    healthy_count += 1
                else:
                    ulcer_count += 1
                sample_idx += 1
        
        print(f"\n✅ Saved {healthy_count + ulcer_count} RGB-only visualizations ({healthy_count} healthy, {ulcer_count} ulcer) to {rgb_output_dir}")
    else:
        print(f"❌ Checkpoint not found: {rgb_checkpoint}")
    
    # =====================
    # THERMAL-ONLY VISUALIZATION
    # =====================
    print("\n" + "🔵 "*35)
    print("VISUALIZING THERMAL-ONLY MODEL")
    print("🔵 "*35)
    
    thermal_checkpoint = CHECKPOINT_DIR / "checkpoints_thermal_only" / "best_model.pt"
    if thermal_checkpoint.exists():
        print(f"Loading: {thermal_checkpoint}")
        thermal_model = ThermalOnlyModel().to(DEVICE)
        if not load_checkpoint_flexible(thermal_model, thermal_checkpoint, DEVICE):
            checkpoint = torch.load(thermal_checkpoint, map_location=DEVICE)
            thermal_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        thermal_model.eval()
        
        thermal_output_dir = OUTPUT_DIR / "thermal_only"
        thermal_output_dir.mkdir(parents=True, exist_ok=True)
        
        healthy_count = 0
        ulcer_count = 0
        for thermal_batch, labels in test_loader_thermal:
            label = labels[0].item()
            
            # Skip if we have enough of this class
            if label == 0 and healthy_count >= num_healthy:
                continue
            if label == 1 and ulcer_count >= num_ulcer:
                continue
            if healthy_count >= num_healthy and ulcer_count >= num_ulcer:
                break
            
            with torch.no_grad():
                fig = visualize_thermal_only(thermal_model, thermal_batch, DEVICE, label)
            
            if fig is not None:
                class_name = 'healthy' if label == 0 else 'ulcer'
                class_count = healthy_count if label == 0 else ulcer_count
                save_path = thermal_output_dir / f"{class_name}_{class_count:02d}.png"
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"  ✓ Saved {save_path.name}")
                
                if label == 0:
                    healthy_count += 1
                else:
                    ulcer_count += 1
        
        print(f"\n✅ Saved {healthy_count + ulcer_count} Thermal-only visualizations ({healthy_count} healthy, {ulcer_count} ulcer) to {thermal_output_dir}")
    else:
        print(f"❌ Checkpoint not found: {thermal_checkpoint}")
    
    # =====================
    # MULTIMODAL VISUALIZATION
    # =====================
    print("\n" + "🟣 "*35)
    print("VISUALIZING MULTIMODAL FUSION MODEL")
    print("🟣 "*35)
    
    multimodal_checkpoint = CHECKPOINT_DIR / "checkpoints_multimodal" / "best_model.pt"
    if multimodal_checkpoint.exists():
        print(f"Loading: {multimodal_checkpoint}")
        multimodal_model = MultimodalFusionModel().to(DEVICE)
        if not load_checkpoint_flexible(multimodal_model, multimodal_checkpoint, DEVICE):
            checkpoint = torch.load(multimodal_checkpoint, map_location=DEVICE)
            multimodal_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        multimodal_model.eval()
        
        multimodal_output_dir = OUTPUT_DIR / "multimodal"
        multimodal_output_dir.mkdir(parents=True, exist_ok=True)
        
        healthy_count = 0
        ulcer_count = 0
        for rgb_batch, thermal_batch, labels in test_loader_multimodal:
            label = labels[0].item()
            
            # Skip if we have enough of this class
            if label == 0 and healthy_count >= num_healthy:
                continue
            if label == 1 and ulcer_count >= num_ulcer:
                continue
            if healthy_count >= num_healthy and ulcer_count >= num_ulcer:
                break
            
            with torch.no_grad():
                fig = visualize_multimodal(multimodal_model, rgb_batch, thermal_batch, 
                                          DEVICE, label)
            
            if fig is not None:
                class_name = 'healthy' if label == 0 else 'ulcer'
                class_count = healthy_count if label == 0 else ulcer_count
                save_path = multimodal_output_dir / f"{class_name}_{class_count:02d}.png"
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"  ✓ Saved {save_path.name}")
                
                if label == 0:
                    healthy_count += 1
                else:
                    ulcer_count += 1
        
        print(f"\n✅ Saved {healthy_count + ulcer_count} Multimodal visualizations ({healthy_count} healthy, {ulcer_count} ulcer) to {multimodal_output_dir}")
    else:
        print(f"❌ Checkpoint not found: {multimodal_checkpoint}")
    
    print("\n" + "="*70)
    print("✅ GRAD-CAM VISUALIZATION COMPLETE")
    print(f"📁 Output directory: {OUTPUT_DIR}")
    print("="*70)


if __name__ == "__main__":
    main()
