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


all_dino = timm.list_models("*dino*")
for m in all_dino:
    print(m)
