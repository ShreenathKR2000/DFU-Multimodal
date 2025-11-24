import torch
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, feat_dim=1280):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)
