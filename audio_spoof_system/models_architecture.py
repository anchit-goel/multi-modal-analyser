import torch
import torch.nn as nn

class SqueezeExcitation(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        scale = self.se(x).view(x.size(0), x.size(1), 1, 1)
        return x * scale


class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.se      = SqueezeExcitation(128)
        self.pool    = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc      = nn.Linear(128, 2)  # 2 classes not 1

    def forward(self, x):
        x = self.features(x)
        x = self.se(x)
        x = self.pool(x).squeeze(-1).squeeze(-1)
        x = self.dropout(x)
        return self.fc(x)  # raw logits, no sigmoid


class MFM(nn.Module):
    """Max Feature Map — splits channels and takes max"""
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return torch.max(x1, x2)


class LCNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 64, 5, padding=2),   MFM(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 1),              MFM(),
            nn.Conv2d(32, 96, 3, padding=1),   MFM(), nn.MaxPool2d(2),
            nn.Conv2d(48, 96, 1),              MFM(),
            nn.Conv2d(48, 128, 3, padding=1),  MFM(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 1),             MFM(),
            nn.Conv2d(64, 64, 3, padding=1),   MFM(),
            nn.Conv2d(32, 64, 1),              MFM(),
        )
        self.pool    = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc      = nn.Linear(32, 2)  # 2 classes not 1

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.pool(x).squeeze(-1).squeeze(-1)
        x = self.dropout(x)
        return self.fc(x)  # raw logits, no sigmoid