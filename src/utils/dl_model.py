from __future__ import annotations

import torch
import torch.nn as nn


class MLPForClassification(nn.Module):
    def __init__(self, input_size: int, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.RMSNorm(512),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.RMSNorm(256),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.RMSNorm(128),
            nn.SiLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)
