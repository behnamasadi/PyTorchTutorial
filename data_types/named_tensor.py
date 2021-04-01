# https://pytorch.org/docs/stable/named_tensor.html
#  Named tensors and all their associated APIs are an experimental feature and subject to change
import torch

imgs = torch.randn(1, 2, 2, 3 , names=('N', 'C', 'H', 'W'))
print(imgs.names)