import torch
import torch.nn as nn

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
        fused = g * rgb_feat + (1 - g) * th_feat
        return fused
