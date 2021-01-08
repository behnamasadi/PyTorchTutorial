import numpy as np
def maxPool2d(H_in,W_in,padding,dilation,kernel_size,stride):
    # input: (N =number of batch , C_in = in_channels, H_in, W_in)
    # output: (N =number of batch , C_out = out_channels, H_out, W_out)
    # https://pytorch.org/docs/master/generated/torch.nn.MaxPool2d.html#torch.nn.MaxPool2d
    H_out=np.floor((H_in+2*padding[0]-dilation[0]*(kernel_size[0]-1)-1)/stride[0]) +1
    W_out=np.floor((W_in+2*padding[1]-dilation[1]*(kernel_size[1]-1)-1)/stride[0]) +1
    return H_out,W_out


def conv2d(H_in,W_in,padding,dilation,kernel_size,stride):
    # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    H_out = np.floor((H_in + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0]) + 1
    W_out = np.floor((W_in + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[0]) + 1
    return H_out, W_out


def avgPool2d(H_in,W_in,padding,kernel_size,stride):
    H_out = np.floor( (H_in + 2 * padding[0] - kernel_size[0] )  / stride[0]) + 1
    W_out = np.floor( (W_in + 2 * padding[1] -kernel_size[1] )  / stride[0]) + 1
    return H_out, W_out