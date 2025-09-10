import torch
import torch.nn as nn
import os
from utility.file_utils import resource_path


class SimpleMNISTModel(nn.Module):
    def __init__(self, num_classes=10, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Feature extraction layers (conv layers) - can be frozen/unfrozen
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32,
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 28x28 -> 14x14

            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 14x14 -> 7x7

            nn.Conv2d(in_channels=64, out_channels=128,
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))  # 7x7 -> 4x4
        )

        # Classifier layers (MLP)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=128*4*4, out_features=256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(in_features=128, out_features=num_classes)
        )

    def forward(self, x):
        # Feature extraction
        x = self.features(x)
        # Classification
        x = self.classifier(x)
        return x


# Create model and test it
myMNIST = SimpleMNISTModel()
# B, C, H, W
input_tensor = torch.randn(1, 1, 28, 28)
print(f"Input shape: {input_tensor.shape}")

# Test forward pass
output = myMNIST(input_tensor)
print(f"Output shape: {output.shape}")

# Save model using resource_path
model_path = resource_path("models", "best_myMNIST.pth")
print("model_path: ", model_path)

# Create models directory if it doesn't exist
model_path.parent.mkdir(parents=True, exist_ok=True)
torch.save(myMNIST.state_dict(), model_path)
print(f"Model saved to: {model_path}")
