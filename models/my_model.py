import torch
import torch.nn as nn


class MyModel(nn.Module):
    """Basic neural network model."""

    def __init__(self, input_size, hidden_size, num_classes, dropout=0.2):
        """
        Initialize the model.

        Args:
            input_size (int): Size of input features
            hidden_size (int): Size of hidden layer
            num_classes (int): Number of output classes
            dropout (float): Dropout probability
        """
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor
        """
        return self.layers(x)
