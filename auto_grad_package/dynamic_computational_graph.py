# https://towardsdatascience.com/pytorch-autograd-understanding-the-heart-of-pytorchs-magic-2686cd94ec95

# The change in the loss for a small change in an input weight is called the gradient of that weight and
# is calculated using backpropagation.
# This is done in an iterative way. For each iteration, several gradients are calculated and something
# called a computation graph is built for storing these gradient functions.
# PyTorch does it by building a Dynamic Computational Graph (DCG).

# This graph is built from scratch in every iteration providing maximum flexibility to gradient calculation.
# For example, for a forward operation (function)Mul a backward operation (function) called MulBackwardis dynamically
# integrated in the backward graph for computing the gradient.

import torch
x=torch.randn(size=[2,3],requires_grad=True)
y=x**2
print(y)