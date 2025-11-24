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
model_name = "*tf_efficientnetv2_s*"

model_name = "*tf_efficientnetv2_l.in21k*"
all_convnext = timm.list_models(model_name)
# Get list of models with pretrained weights available (without downloading)
# Note: pretrained models have variant suffixes (e.g., "convnextv2_tiny.fcmae_ft_in1k")
# while base models don't (e.g., "convnextv2_tiny")
models_with_pretrained = timm.list_models(
    "*tf_efficientnetv2_s*", pretrained=True)

for m in all_convnext:
    # Check if any pretrained variant exists for this base model
    # Pretrained models are either exact match or start with base_name + "."
    has_pretrained = any(
        p == m or p.startswith(m + ".") for p in models_with_pretrained
    )

    if has_pretrained:
        status = "✅"
        print(f"{status} {m}")
    else:
        status = "❌"
        print(f"{status} {m} (no pretrained weights)")


model_name = "convnextv2_tiny"

# Note: convnextv2_small doesn't have pretrained weights
# Option 1: Use pretrained=False (random initialization)
# Option 2: Use a model with pretrained weights like convnextv2_base or convnextv2_large
try:
    model = timm.create_model(model_name, pretrained=True)
except RuntimeError as e:
    print(f"Warning: {e}")
    print(
        f"Creating {model_name} without pretrained weights (pretrained=False)")
    model = timm.create_model(model_name, pretrained=False)


print(model.num_features)


model_config = model.pretrained_cfg
print(model_config)

print(type(model_config["input_size"]))
print("input_size:", model_config["input_size"])
(C, H, W) = model_config["input_size"]

x = torch.randn(1, C, H, W)

out = model(x)
print("output shape:", out.shape)

features = model.forward_features(x)
print("features shape: ", features.shape)


# for name, params in model.named_parameters():
#     print(name, params.shape)
