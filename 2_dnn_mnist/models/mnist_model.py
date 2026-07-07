import torch
import torch.nn as nn
import numpy as np

class MnistNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            # (B,1,28,28)
            nn.Conv2d(1, 32, kernel_size=3, padding=1), # (B,32,28,28)
            nn.ReLU(),
            nn.MaxPool2d(2),              # (B,32,14,14)

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),              # (B,64,7,7)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),                 # (B,3136)
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
