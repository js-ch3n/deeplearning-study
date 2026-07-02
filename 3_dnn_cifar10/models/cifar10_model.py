import torch
import torch.nn as nn


class Cifar10Network(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            # (B,3,32,32)
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),              # (B,32,16,16)

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),              # (B,64,8,8)

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),              # (B,128,4,4)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),                 # (B,2048)
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
