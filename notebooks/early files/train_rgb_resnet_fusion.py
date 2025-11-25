# train_rgb_resnet_fusion.py
"""
Multimodal training: EfficientNet-B0 (RGB) + ResNet-50 (Thermal)
Hybrid (gated) fusion, stratified splits, label-matched random pairing.
Saves best checkpoint to /root/DFU_MMT/logs/checkpoints based on val F1.
"""

import random
import sys
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

# Setup project root and imports (adjust if placed elsewhere)
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

# If you have your own EfficientNetEncoder in models/encoders, import and use it:
try:
    from models.encoders import EfficientNetEncoder
except Exception:
    # Fallback: use torchvision efficientnet_b0
    from torchvision.models import efficientnet_b0
    class EfficientNetEncoder(nn.Module):
        def __init__(self, feat_dim=1280, pretrained=True):
            super().__init__()
            m = efficientnet_b0(weights='DEFAULT' if pretrained else None)
            # keep features, drop classifier
            self.backbone = m.features
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.feat_dim = 1280
        def forward(self, x):
            x = self.backbone(x)
            x = self.pool(x).flatten(1)
            return x

# ---------------------------
# Configuration
# ---------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 12            # adjust if OOM (we used 15 earlier; 12 is safer)
NUM_WORKERS = 4
NUM_EPOCHS = 10
LR = 1e-4
RANDOM_SEED = 42

CHECKPOINT_DIR = project_root / "logs" / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# Data directories (standardized 224x224) - global dataset root
DATA_ROOT = Path("/home/skr/CompVis/Dataset/data")
RGB_DIR = DATA_ROOT / "rgb_standardized"
TH_DIR = DATA_ROOT / "thermal_standardized"

# ---------------------------
# Transforms (stronger augmentations)
# ---------------------------
train_transform_rgb = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

train_transform_th = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

eval_transform_rgb = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

eval_transform_th = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

# ---------------------------
# Helper: list image paths grouped by class
# ---------------------------
def list_images_by_class(root_split_dir):
    # root_split_dir e.g. /.../rgb_standardized/train
    out = defaultdict(list)
    for class_name in ("healthy", "ulcer"):
        p = root_split_dir / class_name
        if p.exists():
            for ext in ('*.jpg','*.jpeg','*.png','*.bmp','*.tif','*.tiff'):
                for f in p.glob(ext):
                    out[class_name].append(f)
    return out

# ---------------------------
# Build label-balanced pairing pool and stratified splits (indices)
# ---------------------------
def build_label_balanced_indices(rgb_base, th_base, seed=RANDOM_SEED):
    # rgb_base, th_base: e.g. RGB_DIR
    # We'll compute minimal per-class count and construct a labels array of length sum(min_counts)
    rgb_train = list_images_by_class(rgb_base / "train")
    rgb_val   = list_images_by_class(rgb_base / "val")
    rgb_test  = list_images_by_class(rgb_base / "test")
    th_train  = list_images_by_class(th_base  / "train")
    th_val    = list_images_by_class(th_base  / "val")
    th_test   = list_images_by_class(th_base  / "test")
    # For each split, we'll create label lists using min count per class between rgb and thermal for that split
    splits = {}
    for split_name, (rgb_map, th_map) in zip(
        ["train","val","test"],
        [(rgb_train, th_train),(rgb_val, th_val),(rgb_test, th_test)]
    ):
        counts = {}
        for cls in ("healthy","ulcer"):
            counts[cls] = min(len(rgb_map[cls]), len(th_map[cls]))
        labels = []
        for cls, cnt in counts.items():
            labels += [0 if cls=="healthy" else 1] * cnt
        # shuffle labels for randomness (we'll use these as targets for pairing)
        random.Random(seed).shuffle(labels)
        splits[split_name] = {
            "rgb_map": rgb_map,
            "th_map": th_map,
            "labels": labels
        }
    return splits

# ---------------------------
# Paired Dataset (label-matched randomized pairing)
# ---------------------------
class PairedDFUDataset(Dataset):
    def __init__(self, rgb_map, th_map, labels, train=True, transform_rgb=None, transform_th=None):
        """
        rgb_map, th_map: dict {'healthy':[paths], 'ulcer':[paths]}
        labels: list of labels (0/1) for dataset length
        Dataset length = len(labels)
        Each __getitem__ returns (rgb_tensor, th_tensor, label)
        For a given label, a random sample from that class is selected from both rgb_map & th_map.
        """
        self.rgb_map = rgb_map
        self.th_map = th_map
        self.labels = labels
        self.train = train
        self.transform_rgb = transform_rgb
        self.transform_th = transform_th
        # Pre-cache lists for faster random choice
        self.rgb_by_label = {0: rgb_map["healthy"], 1: rgb_map["ulcer"]}
        self.th_by_label  = {0: th_map["healthy"], 1: th_map["ulcer"]}
        # If class lists are empty for any class, raise
        for k in (0,1):
            if len(self.rgb_by_label[k]) == 0 or len(self.th_by_label[k]) == 0:
                raise ValueError(f"No images for class {k} in rgb or thermal for this split.")
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        lbl = int(self.labels[idx])
        # choose random sample from class lists
        rgb_path = random.choice(self.rgb_by_label[lbl])
        th_path  = random.choice(self.th_by_label[lbl])
        rgb = Image.open(rgb_path).convert("RGB")
        th  = Image.open(th_path).convert("RGB")
        if self.transform_rgb:
            rgb = self.transform_rgb(rgb)
        if self.transform_th:
            th = self.transform_th(th)
        return rgb, th, torch.tensor(lbl, dtype=torch.float32)

# ---------------------------
# Build splits and DataLoaders
# ---------------------------
random.seed(RANDOM_SEED)
splits = build_label_balanced_indices(RGB_DIR, TH_DIR, seed=RANDOM_SEED)

datasets = {}
dataloaders = {}
for split in ("train","val","test"):
    info = splits[split]
    transform_rgb = train_transform_rgb if split=="train" else eval_transform_rgb
    transform_th  = train_transform_th  if split=="train" else eval_transform_th
    ds = PairedDFUDataset(info["rgb_map"], info["th_map"], info["labels"],
                          train=(split=="train"),
                          transform_rgb=transform_rgb, transform_th=transform_th)
    shuffle = (split=="train")
    bs = BATCH_SIZE if split=="train" else max(8, BATCH_SIZE//2)
    dl = DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=NUM_WORKERS, pin_memory=True)
    datasets[split] = ds
    dataloaders[split] = dl
    print(f"Split {split}: {len(ds)} samples (batch={bs}, shuffle={shuffle})")

# ---------------------------
# Encoders: EfficientNet (RGB) + ResNet50 (Thermal)
# ResNet produces 2048-d features; project to 1280 to match EfficientNet
# ---------------------------
class ThermalResNetEncoder(nn.Module):
    def __init__(self, feat_dim=1280, pretrained=True):
        super().__init__()
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        # remove final fc
        modules = list(m.children())[:-1]  # up to avgpool
        self.backbone = nn.Sequential(*modules)   # outputs (B,2048,1,1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.project = nn.Linear(2048, feat_dim)
        self.feat_dim = feat_dim
    def forward(self, x):
        x = self.backbone(x)              # (B,2048,1,1)
        x = torch.flatten(x, 1)           # (B,2048)
        x = self.project(x)               # (B,feat_dim)
        return x

# Fusion block (gated/hybrid)
class GatedFusion(nn.Module):
    def __init__(self, feat_dim=1280):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim),
            nn.Sigmoid()
        )
    def forward(self, rgb_feat, th_feat):
        combined = torch.cat([rgb_feat, th_feat], dim=1)
        g = self.gate(combined)
        fused = g * rgb_feat + (1.0 - g) * th_feat
        return fused

class ClassifierHead(nn.Module):
    def __init__(self, feat_dim=1280):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.head(x)

# instantiate models
rgb_encoder = EfficientNetEncoder().to(DEVICE)
th_encoder  = ThermalResNetEncoder().to(DEVICE)
fusion = GatedFusion().to(DEVICE)
classifier = ClassifierHead().to(DEVICE)

# collect params, freeze lower layers initially (recommended)
# freeze most of both backbones first, fine-tune later if you want
for p in rgb_encoder.parameters():
    p.requires_grad = True  # leave True for small dataset fine-tuning; set False to freeze
# optionally freeze some of ResNet layers:
# for name,param in th_encoder.named_parameters():
#     if 'layer4' not in name:
#         param.requires_grad = False

params = list(rgb_encoder.parameters()) + list(th_encoder.parameters()) + list(fusion.parameters()) + list(classifier.parameters())

optimizer = AdamW(params, lr=LR, weight_decay=1e-4)
criterion = nn.BCELoss()

# ---------------------------
# Training loop with val/test + best model saving by val F1
# ---------------------------
best_val_f1 = 0.0
best_path = None

for epoch in range(1, NUM_EPOCHS + 1):
    # training
    rgb_encoder.train(); th_encoder.train(); fusion.train(); classifier.train()
    train_losses = []
    train_preds = []
    train_labels = []
    for rgb_batch, th_batch, labels in dataloaders["train"]:
        rgb_batch = rgb_batch.to(DEVICE)
        th_batch  = th_batch.to(DEVICE)
        labels    = labels.to(DEVICE)

        optimizer.zero_grad()
        rgb_feat = rgb_encoder(rgb_batch)    # (B,1280)
        th_feat  = th_encoder(th_batch)      # (B,1280)
        fused    = fusion(rgb_feat, th_feat)
        out      = classifier(fused).squeeze()
        loss     = criterion(out, labels)
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())
        train_preds.append(torch.round(out.detach().cpu()))
        train_labels.append(labels.detach().cpu())

    train_preds = torch.cat(train_preds)
    train_labels = torch.cat(train_labels)
    train_acc = accuracy_score(train_labels, train_preds)
    train_f1  = f1_score(train_labels, train_preds)
    train_loss = sum(train_losses)/len(train_losses)

    # validation
    rgb_encoder.eval(); th_encoder.eval(); fusion.eval(); classifier.eval()
    val_losses = []
    val_preds = []
    val_labels = []

    with torch.no_grad():
        for rgb_batch, th_batch, labels in dataloaders["val"]:
            rgb_batch = rgb_batch.to(DEVICE)
            th_batch  = th_batch.to(DEVICE)
            labels    = labels.to(DEVICE)
            rgb_feat = rgb_encoder(rgb_batch)
            th_feat  = th_encoder(th_batch)
            fused    = fusion(rgb_feat, th_feat)
            out      = classifier(fused).squeeze()
            loss     = criterion(out, labels)

            val_losses.append(loss.item())
            val_preds.append(torch.round(out.detach().cpu()))
            val_labels.append(labels.detach().cpu())

    val_preds = torch.cat(val_preds)
    val_labels = torch.cat(val_labels)
    val_acc = accuracy_score(val_labels, val_preds)
    val_f1  = f1_score(val_labels, val_preds)
    val_loss = sum(val_losses)/len(val_losses)

    print(f"[Epoch {epoch}/{NUM_EPOCHS}] "
          f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f} | "
          f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")

    # save checkpoint if best val F1
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_path = CHECKPOINT_DIR / f"best_epoch_{epoch}_valF1_{best_val_f1:.4f}.pt"
        torch.save({
            "epoch": epoch,
            "rgb_state": rgb_encoder.state_dict(),
            "th_state": th_encoder.state_dict(),
            "fusion_state": fusion.state_dict(),
            "classifier_state": classifier.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_f1": best_val_f1
        }, best_path)
        print(f"💾 Saved NEW BEST model -> {best_path}")

# ---------------------------
# Final test evaluation (load best model if exists)
# ---------------------------
if best_path is not None:
    ckpt = torch.load(best_path, map_location=DEVICE)
    rgb_encoder.load_state_dict(ckpt["rgb_state"])
    th_encoder.load_state_dict(ckpt["th_state"])
    fusion.load_state_dict(ckpt["fusion_state"])
    classifier.load_state_dict(ckpt["classifier_state"])
    print(f"Loaded best model from {best_path} for final test evaluation.")

rgb_encoder.eval(); th_encoder.eval(); fusion.eval(); classifier.eval()
test_losses = []; test_preds=[]; test_labels=[]
with torch.no_grad():
    for rgb_batch, th_batch, labels in dataloaders["test"]:
        rgb_batch = rgb_batch.to(DEVICE); th_batch = th_batch.to(DEVICE); labels = labels.to(DEVICE)
        rgb_feat = rgb_encoder(rgb_batch); th_feat = th_encoder(th_batch)
        fused = fusion(rgb_feat, th_feat); out = classifier(fused).squeeze()
        loss = criterion(out, labels)
        test_losses.append(loss.item())
        test_preds.append(torch.round(out.detach().cpu()))
        test_labels.append(labels.detach().cpu())

test_preds = torch.cat(test_preds); test_labels = torch.cat(test_labels)
test_acc = accuracy_score(test_labels, test_preds); test_f1 = f1_score(test_labels, test_preds)
test_loss = sum(test_losses)/len(test_losses)
print("\n" + "="*60)
print(f"TEST Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, F1: {test_f1:.4f}")
