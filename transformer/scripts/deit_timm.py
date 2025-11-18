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


all_deit = timm.list_models("*deit*")
for m in all_deit:
    print(m)
# deit_base_patch16_224 = timm.create_model("deit_base_patch16_224", pretrained=True)
