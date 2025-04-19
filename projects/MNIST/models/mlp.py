import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_dim=28 * 28, hidden_dim=256, output_dim=10, dropout=0.5):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.dropout = dropout

        # First layer with batch normalization
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        # Second layer with batch normalization
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.bn2 = nn.BatchNorm1d(output_dim)

    def forward(self, x):
        # Flatten the input
        x = x.view(x.size(0), -1)

        # First layer with batch norm and ReLU
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Second layer with batch norm
        x = self.fc2(x)
        x = self.bn2(x)

        return x
