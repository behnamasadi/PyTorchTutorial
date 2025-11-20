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


all_RegNetY = timm.list_models("*regnety*")
for m in all_RegNetY:
    print(m)

model_name = "regnety_032"
model = timm.create_model(model_name, pretrained=True, features_only=True)

cfg = model.pretrained_cfg
print(cfg)

print("*"*60)
# print(model)
print("*"*60)
# for stage in model.nam:
#     print(stage)


for name, module in model.named_children():
    print(f"  {name}: {type(module).__name__}")


print("-"*60)

name, module = list(model.named_children())[1]
print(module)
print("-"*60)


x = torch.randn(1, 3, 224, 224)
features = model(x)
for f in features:
    print(f.shape)
