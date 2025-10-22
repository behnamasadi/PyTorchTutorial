import torch
x = torch.randn(2, 1)
print(x)

x_expand = x.expand(-1, 3)
# print()
x_expand[0, 0] = x_expand[0, 0]+1
print(x_expand)
