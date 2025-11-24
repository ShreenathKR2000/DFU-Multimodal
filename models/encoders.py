import torch
import torch.nn as nn
from torchvision import models

class EfficientNetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights="IMAGENET1K_V1")
        self.backbone.classifier = nn.Identity()   # 1280-d features

    def forward(self, x):
        return self.backbone(x)
