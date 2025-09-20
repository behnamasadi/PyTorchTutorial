
import torch
import torch.nn as nn

# Input: 1x1x2x2
x = torch.tensor([[[[1., 2.],
                    [3., 4.]]]])

# ConvTranspose2d layer: 1 input channel, 1 output channel, kernel 2x2, stride 2
deconv = nn.ConvTranspose2d(1, 1, kernel_size=2, stride=2, bias=False)

# Set custom weights to match our manual example
with torch.no_grad():
    deconv.weight.copy_(torch.tensor([[[[1., 0.],
                                        [0., 1.]]]]))

# Apply layer
y = deconv(x)

print("Input:\n", x)
print("Weights:\n", deconv.weight)
print("Output:\n", y)




import torch
import torch.nn as nn

deconv1 = nn.ConvTranspose2d(1, 1, kernel_size=3, stride=2)
deconv2 = nn.ConvTranspose2d(1, 1, kernel_size=3, stride=2)

print(id(deconv1.weight), id(deconv2.weight))
