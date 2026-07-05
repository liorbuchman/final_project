import torch
import torch.nn as nn

class SmallCNN(nn.Module):
    """Simple CNN for log-mel spectrogram classification.

    Input:  [B, 1, MELS, TIME]
    Output: logits [B, n_classes]
    """
    def __init__(self, n_classes: int):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), #can change the kernel sizw to 5x5
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # Global Average Pooling
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.feat(x)
        z = self.head(z)
        return z
