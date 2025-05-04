import torch
import torch.nn as nn


layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
print(layer.self_attn)
