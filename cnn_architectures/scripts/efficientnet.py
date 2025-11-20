# fmt: off
# isort: skip_file
# DO NOT reorganize imports - warnings filter must be FIRST!

import torch.nn.functional as F
import torch
import warnings
import os

warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

import timm
# fmt: on

# ≤ 4 GB VRAM	tf_efficientnetv2_b0.in1k	Small, accurate, lightweight
# 4–8 GB	tf_efficientnetv2_s.in21k_ft_in1k	Balanced, fast training
# 8–16 GB	tf_efficientnetv2_m.in21k_ft_in1k	Higher accuracy
# ≥ 16 GB	tf_efficientnetv2_l.in21k_ft_in1k	SOTA accuracy
# > 24 GB	tf_efficientnetv2_xl.in21k_ft_in1k	Maximum accuracy, expensive
# all_efficientnet = timm.list_models("*efficientnet*", pretrained=True)
# for m in all_efficientnet:
#     print(m)


model_name = "tf_efficientnetv2_s.in21k_ft_in1k"
efficientnetv2_s = timm.create_model(model_name, pretrained=True)

print("=" * 60)
print("MODEL ARCHITECTURE")
print("=" * 60)

# Print high-level structure
print("\nMain components:")
for name, module in efficientnetv2_s.named_children():
    print(f"  {name}: {type(module).__name__}")

print("\n" + "=" * 60)
print("DETAILED COMPONENTS")
print("=" * 60)

# Access specific parts
print(f"\n1. Stem (initial conv): {efficientnetv2_s.conv_stem}")
print(f"   Output channels: {efficientnetv2_s.conv_stem.out_channels}")

print(f"\n2. Blocks (main backbone):")
for i, block in enumerate(efficientnetv2_s.blocks):
    print(f"   Block {i}: {type(block).__name__}")

print(f"\n3. Head (classifier):")
print(f"   Global pooling: {efficientnetv2_s.global_pool}")
print(f"   Classifier: {efficientnetv2_s.classifier}")
if hasattr(efficientnetv2_s.classifier, 'in_features'):
    print(f"   - Input features: {efficientnetv2_s.classifier.in_features}")
    print(f"   - Output classes: {efficientnetv2_s.classifier.out_features}")

print("\n" + "=" * 60)
print("REPLACING HEAD FOR FINE-TUNING")
print("=" * 60)

# Get number of input features to the classifier
num_features = efficientnetv2_s.classifier.in_features
num_classes = 10  # Your custom number of classes

# Method 1: Replace the classifier head
efficientnetv2_s.classifier = torch.nn.Linear(num_features, num_classes)
print(f"\n✅ Replaced head: {num_features} -> {num_classes} classes")
print(f"   New classifier: {efficientnetv2_s.classifier}")

print("\n" + "=" * 60)
print("FREEZING BACKBONE FOR FINE-TUNING")
print("=" * 60)

# Freeze all layers except the head
for name, param in efficientnetv2_s.named_parameters():
    if 'classifier' not in name:
        param.requires_grad = False
    else:
        param.requires_grad = True

# Count trainable parameters
trainable = sum(p.numel()
                for p in efficientnetv2_s.parameters() if p.requires_grad)
total = sum(p.numel() for p in efficientnetv2_s.parameters())
print(
    f"\nTrainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

# Test forward pass
x = torch.randn(2, 3, 224, 224)
output = efficientnetv2_s(x)
print(f"\nTest forward pass: {x.shape} -> {output.shape}")
