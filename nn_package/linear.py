import torch
import torch.nn as nn

# Applies a linear transformation to the incoming data: y = xA^T + b

# 3x2
x = torch.tensor([[1.0, -4.0],
                  [0.1,  2.0],
                  [7.0,  1.0],
                  [2.0,  5.0]])

in_features = x.shape[1]
out_features = 3

model = nn.Linear(in_features, out_features,bias=True)

print(model.weight)
print(model.bias)
