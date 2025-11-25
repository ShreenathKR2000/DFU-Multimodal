# train_multimodal_with_val_and_test.py
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score

# Add DFU_MMT root to sys.path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from scripts.dataloader import DFUDataset, get_transforms
from models.encoders import EfficientNetEncoder
from torch.utils.data import DataLoader

# =====================
# CONFIGURATION
# =====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 15
NUM_WORKERS = 4
NUM_EPOCHS = 10
LR = 1e-4

CHECKPOINT_DIR = project_root / "logs" / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# =====================
# TRANSFORMS
# =====================
rgb_transforms = get_transforms("rgb")
thermal_transforms = get_transforms("thermal")

# =====================
# DATASETS & DATALOADERS
# =====================
# Use the global standardized dataset location
data_dir = Path("/home/skr/CompVis/Dataset/data")
rgb_dir = data_dir / "rgb_standardized"
thermal_dir = data_dir / "thermal_standardized"

datasets = {}
dataloaders = {}

for split in ["train", "val", "test"]:
    datasets[split] = DFUDataset(
        rgb_dir=rgb_dir,
        thermal_dir=thermal_dir,
        split=split,
        transform_rgb=rgb_transforms,
        transform_thermal=thermal_transforms
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
class MultimodalClassifier(nn.Module):
    def __init__(self, feat_dim=1280):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.Sigmoid()
        )
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, rgb_feat, thermal_feat):
        fused_input = torch.cat([rgb_feat, thermal_feat], dim=1)
        gate = self.gate(fused_input)
        fused = gate * rgb_feat + (1 - gate) * thermal_feat
        return self.classifier(fused)

# Instantiate encoders and fusion classifier
rgb_encoder = EfficientNetEncoder().to(DEVICE)
thermal_encoder = EfficientNetEncoder().to(DEVICE)
fusion_model = MultimodalClassifier().to(DEVICE)

# Loss & optimizer
criterion = nn.BCELoss()
optimizer = AdamW(
    list(rgb_encoder.parameters()) + 
    list(thermal_encoder.parameters()) + 
    list(fusion_model.parameters()),
    lr=LR,
    weight_decay=1e-4
)

# =====================
# TRAIN + VALIDATION LOOP
# =====================
for epoch in range(1, NUM_EPOCHS + 1):
    rgb_encoder.train()
    thermal_encoder.train()
    fusion_model.train()

    running_loss = 0
    all_preds, all_labels = [], []

    for rgb_batch, thermal_batch, labels in dataloaders["train"]:
        rgb_batch = rgb_batch.to(DEVICE)
        thermal_batch = thermal_batch.to(DEVICE)
        labels = labels.float().to(DEVICE)

        optimizer.zero_grad()
        rgb_feat = rgb_encoder(rgb_batch)
        thermal_feat = thermal_encoder(thermal_batch)
        output = fusion_model(rgb_feat, thermal_feat).squeeze()
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        all_preds.append(torch.round(output).detach().cpu())
        all_labels.append(labels.detach().cpu())

    # Train metrics
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    train_acc = accuracy_score(all_labels, all_preds)
    train_f1 = f1_score(all_labels, all_preds)
    train_loss = running_loss / len(dataloaders["train"])

    # =====================
    # Validation
    # =====================
    rgb_encoder.eval()
    thermal_encoder.eval()
    fusion_model.eval()
    val_loss = 0
    val_preds, val_labels = [], []

    with torch.no_grad():
        for rgb_batch, thermal_batch, labels in dataloaders["val"]:
            rgb_batch = rgb_batch.to(DEVICE)
            thermal_batch = thermal_batch.to(DEVICE)
            labels = labels.float().to(DEVICE)

            rgb_feat = rgb_encoder(rgb_batch)
            thermal_feat = thermal_encoder(thermal_batch)
            output = fusion_model(rgb_feat, thermal_feat).squeeze()
            loss = criterion(output, labels)

            val_loss += loss.item()
            val_preds.append(torch.round(output).detach().cpu())
            val_labels.append(labels.detach().cpu())

    val_preds = torch.cat(val_preds)
    val_labels = torch.cat(val_labels)
    val_acc = accuracy_score(val_labels, val_preds)
    val_f1 = f1_score(val_labels, val_preds)
    val_loss = val_loss / len(dataloaders["val"])

    print(f"[Epoch {epoch}/{NUM_EPOCHS}] "
          f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f} | "
          f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")

    # Save checkpoint
    checkpoint_path = CHECKPOINT_DIR / f"epoch_{epoch}.pt"
    torch.save({
        "epoch": epoch,
        "rgb_encoder_state": rgb_encoder.state_dict(),
        "thermal_encoder_state": thermal_encoder.state_dict(),
        "fusion_model_state": fusion_model.state_dict(),
        "optimizer_state": optimizer.state_dict()
    }, checkpoint_path)
    print(f"✅ Checkpoint saved at {checkpoint_path}")

# =====================
# TEST EVALUATION
# =====================
print("\n" + "="*60)
print("EVALUATING ON TEST SPLIT")
print("="*60)

rgb_encoder.eval()
thermal_encoder.eval()
fusion_model.eval()
test_loss = 0
test_preds, test_labels = [], []

with torch.no_grad():
    for rgb_batch, thermal_batch, labels in dataloaders["test"]:
        rgb_batch = rgb_batch.to(DEVICE)
        thermal_batch = thermal_batch.to(DEVICE)
        labels = labels.float().to(DEVICE)

        rgb_feat = rgb_encoder(rgb_batch)
        thermal_feat = thermal_encoder(thermal_batch)
        output = fusion_model(rgb_feat, thermal_feat).squeeze()
        loss = criterion(output, labels)

        test_loss += loss.item()
        test_preds.append(torch.round(output).detach().cpu())
        test_labels.append(labels.detach().cpu())

test_preds = torch.cat(test_preds)
test_labels = torch.cat(test_labels)
test_acc = accuracy_score(test_labels, test_preds)
test_f1 = f1_score(test_labels, test_preds)
test_loss = test_loss / len(dataloaders["test"])

print(f"\nTest Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, F1: {test_f1:.4f}")
