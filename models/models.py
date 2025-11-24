import torch
import torch.nn as nn
from torchvision.models import resnet50
from timm import create_model  # for ViT

class RGBResNetEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.resnet = resnet50(pretrained=pretrained)
        self.resnet.fc = nn.Identity()  # remove classifier

    def forward(self, x):
        return self.resnet(x)  # (batch, 2048)

class ThermalViTEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.vit = create_model('vit_base_patch16_224', pretrained=pretrained)
        self.vit.head = nn.Identity()  # remove classifier

    def forward(self, x):
        return self.vit(x)  # (batch, 768)

class MultimodalFusion(nn.Module):
    """
    Late concatenation + MLP classifier
    """
    def __init__(self, rgb_dim=2048, thermal_dim=768, hidden_dim=512):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(rgb_dim + thermal_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, rgb_feat, thermal_feat):
        fused = torch.cat([rgb_feat, thermal_feat], dim=1)
        return self.classifier(fused)
