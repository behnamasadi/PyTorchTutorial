# https://discuss.pytorch.org/t/valueerror-cant-optimize-a-non-leaf-tensor/21751
# A leaf Variable is a variable that is at the beginning of the graph. That means that no operation tracked by the autograd engine created it.
# This is what you want when you optimize neural networks as it is usually your weights or input.

#  autograd keeps a record of data (tensors) & all executed operations
#  (along with the resulting new tensors) in a directed acyclic graph (DAG) consisting of
#  Function objects. In this DAG, leaves are the input tensors, roots are the output tensors.

import torch

# a is a leaf variable
a = torch.rand(10, requires_grad=True)
print(a.is_leaf)

# a is NOT a leaf variable as it was created by the operation that cast a float tensor into a double tensor
a = torch.rand(10, requires_grad=True).double()
print(a.is_leaf)

# equivalent to the formulation just above: not a leaf variable
a = torch.rand(10).requires_grad_().double()
print(a.is_leaf)

# a does not require gradients and has not operation creating it (tracked by the autograd engine).
a = torch.rand(10).double()
print(a.is_leaf)

# a requires gradients and has no operations creating it: it's a leaf variable and can be given to an optimizer.
a = torch.rand(10).double().requires_grad_()
print(a.is_leaf)

# a requires grad, has not operation creating it: it's a leaf variable as well and can be given to an optimizer
#a = torch.rand(10, requires_grad=True, device="cuda")