import torch
import torch.nn as nn


in_channels = 4
out_channels = 5
width = 5
height = 5
batch_size = 1
input = torch.randn(batch_size, in_channels, height, width)
print(input.shape)

# Reshape input to 2D for linear layer
input_reshaped = input.view(batch_size, -1)  # Flatten spatial dimensions
print(f"Reshaped input shape: {input_reshaped.shape}")

# Linear layer expects flattened input
fc = nn.Linear(in_features=in_channels * height * width,
               out_features=out_channels, bias=True)
fc_output = fc(input_reshaped)
print(f"FC output shape: {fc_output.shape}")

# Equivalent Conv2d layer
conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                 kernel_size=(height, width), bias=True)
conv_output = conv(input)
print(f"Conv output shape: {conv_output.shape}")
