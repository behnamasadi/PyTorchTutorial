
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


print("=" * 20, "All available PVT Models ", "=" * 20,)
all_pvt = timm.list_models("*pvt*")
for m in all_pvt:
    print(m)


pvt_model_name = "pvt_v2_b0"

pvt_model = timm.create_model("pvt_v2_b0", pretrained=True)


print("=" * 60)
print(f"MODEL ARCHITECTURE: {pvt_model_name}")
print("=" * 60)

# Print high-level structure
print("\nMain components:")
for name, module in pvt_model.named_children():
    print(f"  {name}: {type(module).__name__}")


print("\n" + "=" * 60)
print("DETAILED COMPONENTS")
print("=" * 60)

# Access specific parts
print(f"\n1. OverlapPatchEmbed: {pvt_model.patch_embed}")
print(f"   Output channels: {pvt_model.patch_embed.proj.out_channels}")


# print(f"FULL ARCHITECTURE: {pvt_model}")


print(f"\n2. Stages (main backbone):")
for i, stage in enumerate(pvt_model.stages):
    print(f"   Stage {i}: {type(stage).__name__}", )


print("=" * 60)
print(f"   Details of Stage 0: ", pvt_model.stages[0])
print("=" * 60)


print(f"\n3. Head (classifier):")
print(f"   Head: {pvt_model.head}")
if hasattr(pvt_model.head, 'in_features'):
    print(f"   - Input features: {pvt_model.head.in_features}")
    print(f"   - Output classes: {pvt_model.head.out_features}")


print("\n" + "=" * 60)
print("REPLACING HEAD FOR FINE-TUNING")
print("=" * 60)

# Get number of input features to the classifier
num_features = pvt_model.head.in_features
num_classes = 10  # Your custom number of classes

# Method 1: Replace the classifier head
pvt_model.head = torch.nn.Linear(num_features, num_classes)
print(f"\n✅ Replaced head: {num_features} -> {num_classes} classes")
print(f"   New classifier: {pvt_model.head}")


print("\n" + "=" * 60)
print("FREEZING BACKBONE FOR FINE-TUNING")
print("=" * 60)


# Freeze all layers except the head
for name, param in pvt_model.named_parameters():
    if 'head' not in name:
        param.requires_grad = False
    else:
        param.requires_grad = True


# Count trainable parameters
trainable = sum(p.numel()
                for p in pvt_model.parameters() if p.requires_grad)
total = sum(p.numel() for p in pvt_model.parameters())
print(
    f"\nTrainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")


print("\n" + "=" * 60)
print("PASSING INPUT THROUGH THE MODEL")
print("=" * 60)


cfg = pvt_model.pretrained_cfg  # (older timm versions call it default_cfg)
print("Required input size:", cfg["input_size"])  # e.g., (3, 224, 224)

# Random input (B=1, C=3, H=224, W=224)
x = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    y = pvt_model(x)

print("Output shape:", y.shape)


print("\n" + "=" * 60)
print("FEATURES ONLY MODEL")
print("=" * 60)

pvt_model_features_only = timm.create_model(
    'pvt_v2_b2', pretrained=True, features_only=True)
with torch.no_grad():
    features = pvt_model_features_only(x)


for i, f in enumerate(features):
    print(f"Stage {i+1} feature:", f.shape)


# A **complete minimal segmentation example** using `pvt_v2_b2` from timm + a small decoder (like a U-Net or FPN-style head)
print("\n" + "=" * 60)
print("COMPLETE MINIMAL SEGMENTATION EXAMPLE")
print("=" * 60)
