import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))   # add DFU_MMT root, NOT scripts/models

import torch
import torch.nn as nn
from torch.optim import AdamW

# Now absolute imports work
from scripts.dataloader import get_dataloaders
from models.encoders import EfficientNetEncoder
from models.fusion import GatedFusion
from models.classifier import Classifier

# === DEVICE SETUP ===
device = "cuda" if torch.cuda.is_available() else "cpu"

# === LOAD DATA ===
# Use the global standardized dataset location
dataloaders, _ = get_dataloaders(
    data_dir=Path("/home/skr/CompVis/Dataset/data"),
    batch_size=12,
    num_workers=4
)

# === MODELS ===
rgb_encoder = EfficientNetEncoder().to(device)
thermal_encoder = EfficientNetEncoder().to(device)
fusion = GatedFusion().to(device)
classifier = Classifier().to(device)

# === LOSS ===
criterion = nn.BCELoss()

# === OPTIMIZER ===
params = list(rgb_encoder.parameters()) + \
         list(thermal_encoder.parameters()) + \
         list(fusion.parameters()) + \
         list(classifier.parameters())

optimizer = AdamW(params, lr=1e-4, weight_decay=1e-4)

# === TRAIN LOOP ===
for epoch in range(10):
    rgb_encoder.train()
    thermal_encoder.train()
    fusion.train()
    classifier.train()

    running_loss = 0

    for rgb, thermal, labels in dataloaders["train"]:
        rgb = rgb.to(device)
        thermal = thermal.to(device)
        labels = labels.float().to(device)

        optimizer.zero_grad()

        rgb_feat = rgb_encoder(rgb)
        th_feat = thermal_encoder(thermal)
        fused = fusion(rgb_feat, th_feat)
        out = classifier(fused).squeeze()

        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}: Loss = {running_loss/len(dataloaders['train']):.4f}")
