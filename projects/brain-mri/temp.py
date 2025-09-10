import torch

gen=torch.Generator().manual_seed(42)

num=torch.randn(size=[10],generator=gen )
# num=torch.randn(10 )
print(num)
# print(num.shape)
print(torch.mean(num))