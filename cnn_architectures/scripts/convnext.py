# fmt: off
# isort: skip_file
# DO NOT reorganize imports - warnings filter must be FIRST!

import torch.nn.functional as F
import torch
import torch.nn as nn
import warnings
import os

warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

import timm
# fmt: on


all_convnext = timm.list_models("*convnextv2*")
for m in all_convnext:
    print(m)
exit()

# model_name = "convnextv2_base"
# model_name = "convnextv2_small"
model_name = "convnextv2_base"

model = timm.create_model(model_name, pretrained=True)

print("-"*60)

for name, params in model.named_parameters():
    print(name)

print("*"*60)

for name, module in model.named_children():
    print(f"  {name}: {type(module).__name__}")

print("*"*60)
cfg = model.pretrained_cfg
print(cfg)

print("*"*60)
print("Understanding the output to the head:")
print("*"*60)

# Create a dummy input
x = torch.randn(1, 3, 224, 224)

# Get the output that goes into the head (this is what you need to know)
with torch.no_grad():
    # forward_features returns the output after stages.3.blocks.2
    # This is the LAST part of the network before the head
    features = model.forward_features(x)

    print(f"Output to head (from stages.3.blocks.2): {features.shape}")
    print(f"  - Channels: {features.shape[1]}")
    print(f"  - Spatial size: {features.shape[2]}x{features.shape[3]}")
    print(f"  - This is the output from: stages.3.blocks.2 (last block)")
    print()
    print(f"Head expects: {model.num_features} channels")
    print()
    print("Key information for replacing the head:")
    print(
        f"  ✅ Input shape: {features.shape} = [batch, channels, height, width]")
    print(
        f"  ✅ Input channels: {features.shape[1]} (use model.num_features: {model.num_features})")
    print(f"  ✅ Input spatial size: {features.shape[2]}x{features.shape[3]}")
    print()
    print("Two approaches for the head input:")
    flattened_size = features.shape[1] * features.shape[2] * features.shape[3]
    print(
        f"  1. With global pooling: [B, {features.shape[1]}, {features.shape[2]}, {features.shape[3]}] -> pool -> [B, {features.shape[1]}]")
    print(
        f"  2. Without pooling (flatten all): [B, {features.shape[1]}, {features.shape[2]}, {features.shape[3]}] -> flatten(1) -> [B, {flattened_size}]")
    print(
        f"     (where {flattened_size} = {features.shape[1]} * {features.shape[2]} * {features.shape[3]} = 7*7*768)")
    print()
    print("The parameters you listed are:")
    print("  - stages.3.blocks.2.* -> This is the LAST block before head")
    print("  - head.norm.* -> Normalization in the head")
    print("  - head.fc.* -> Final classification layer")
    print()
    print("To replace the head, you need the output from stages.3.blocks.2")
    print("which has shape:", features.shape)

print("*"*60)
print("Example: Replace head with custom head")
print("*"*60)

# Get the number of features from the model
num_features = model.num_features

# Calculate flattened size if not using global pooling
with torch.no_grad():
    features = model.forward_features(x)
    flattened_size = features.shape[1] * features.shape[2] * features.shape[3]
    print(f"Input shape to head: {features.shape}")
    print(f"Two options for custom head:")
    print(f"  1. With global pooling: flatten to {num_features} features")
    print(
        f"  2. Without global pooling: flatten to {flattened_size} features (7*7*768)")

print()

num_classes = 10

# Option 1: With global average pooling (like original head)
# This reduces spatial dimensions first, then flattens
custom_head_v1_a = nn.Sequential(
    # Global average pooling: [B, 768, 7, 7] -> [B, 768, 1, 1]
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten(1),  # [B, 768, 1, 1] -> [B, 768] (flatten from dim 1)
    # [B, 768] -> [B, 10]
    nn.Linear(in_features=num_features, out_features=num_classes)
)
print("Option 1: With global average pooling")
print(f"  Input: {features.shape} -> Pool -> Flatten -> {num_features} features -> Linear -> 10 classes")


custom_head_v1_b = nn.Sequential(
    # Global average pooling: [B, 768, 7, 7] -> [B, 768, 1, 1]
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten(1),  # [B, 768, 1, 1] -> [B, 768] (flatten from dim 1)
    nn.LayerNorm(768),
    nn.Dropout(0.3),
    # [B, 768] -> [B, 10]
    nn.Linear(in_features=num_features,  out_features=128),
    nn.ReLU(),
    nn.Dropout(0.25),
    nn.Linear(128, out_features=num_classes)
)
print("Option 1: With global average pooling")
print(f"  Input: {features.shape} -> Pool -> Flatten -> {num_features} features -> Linear -> 10 classes")


# Option 2: Without global pooling (flatten all spatial dimensions)
# This flattens the entire feature map
custom_head_v2 = nn.Sequential(
    nn.Flatten(1),  # [B, 768, 7, 7] -> [B, 768*7*7] = [B, 37632]
    nn.Linear(flattened_size, 10)  # [B, 37632] -> [B, 10]
)
print(f"Option 2: Without global pooling (flatten all spatial dims)")
print(
    f"  Input: {features.shape} -> Flatten -> {flattened_size} features -> Linear -> 10 classes")

# Replace with option 2 (the one you're asking about)
model.head = custom_head_v2

print()
print(f"Using Option 2: Flattening all spatial dimensions")
print(f"Custom head input: {flattened_size} features (7*7*768)")

# Test the new head
with torch.no_grad():
    features = model.forward_features(x)
    print(f"Features shape before head: {features.shape}")
    output = model.head(features)
    print(f"Output shape after custom head: {output.shape}")
