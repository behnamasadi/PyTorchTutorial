import torch
import sys
sys.path.append("../utility")
import shape_calculator

height = 28
width = 28

# default parms are
# stride=None ==> stride = (1, 1)
# padding=0 ==> padding = (0, 0)
# dilation=1 ==> dilation = (1, 1)

kernel_size = (2, 2)
stride = (1, 1)
padding = (0, 0)
dilation = (1, 1)

print("############################### MaxPool2d ############################### ")

max_pooling=torch.nn.MaxPool2d(kernel_size=kernel_size,stride=stride,padding=padding,dilation=dilation)


channels=1
number_of_batch=20
height=28
width=28
input = torch.randn(number_of_batch, channels, height, width)

print("input.shape:", input.shape)

output=max_pooling(input)
print("output.shape:", output.shape)
print("calculated height and width:",shape_calculator.maxPool2d(height,width,padding,dilation,kernel_size,stride))

# torch.nn.functional.max_pool2d() Examples
# https://www.programcreek.com/python/example/104444/torch.nn.functional.max_pool2d




print("############################### AdaptiveAvgPool2d ###############################")
# In average-pooling or max-pooling, you essentially set the stride and kernel-size by your own, setting them
# as hyper-parameters. You will have to re-configure them if you happen to change your input size.
#
# In Adaptive Pooling on the other hand, we specify the output size instead. And the stride and kernel-size
# are automatically selected to adapt to the needs.

input=torch.randn(number_of_batch, channels, height, width)
h_out=10
w_out=10
m=torch.nn.AdaptiveAvgPool2d((h_out,w_out))
output=m(input)
print(output.shape)

print("############################### AvgPool2d ############################### ")

m=torch.nn.AvgPool2d(kernel_size=kernel_size,stride=stride,padding=padding)
input = torch.randn(number_of_batch, channels, height, width)
output=m(input)
print(output.shape)
print(shape_calculator.avgPool2d(height, width,kernel_size=kernel_size,stride=stride,padding=padding))