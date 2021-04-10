# https://pytorch.org/docs/stable/named_tensor.html
#  Named tensors and all their associated APIs are an experimental feature and subject to change
import torch

imgs = torch.randn(1, 2, 2, 3 , names=('N', 'C', 'H', 'W'))
print(imgs.names)


renamed_imgs = imgs.rename(H='height', W='width')
print(renamed_imgs.names)

imgs = torch.randn(1, 2, 2, 3 , names=(None, 'C', 'H', 'W'))


# Two names match if they are equal (string equality) or if at least one is None

x = torch.randn(3, names=('X',))
y = torch.randn(3)
z = torch.randn(3, names=('Z',))

x + y
# error
#x + z
x + x

# https://pytorch.org/docs/stable/named_tensor.html#named-tensors-doc