import torch

print("############################# torch.autograd #############################")
# torch.autograd is PyTorch’s automatic differentiation engine
x=torch.tensor([2.0 ,3.0],requires_grad=True)
y=torch.tensor([6.0 ,4.0],requires_grad=True)
z=3*x**3-y**2

# When we call .backward() on z, autograd calculates these gradients and stores them in the respective tensors’ .grad attribute.
# We need to explicitly pass a gradient argument in z.backward() because it is a vector.
# gradient is a tensor of the same shape as Q, and it represents the gradient of z w.r.t. itself, i.e. dz\dz=1
external_grad=torch.empty(2,requires_grad=True)
z.backward(external_grad)

print(external_grad.grad)

# Gradients are now deposited in x.grad and y.grad
print(x.grad)
print(x.shape)
print(y.grad)

print("############################# Computational Graph #############################")
# autograd keeps a record of data (tensors) & all executed operations (along with the resulting new tensors)
# in a directed acyclic graph (DAG)
# leaves are the input tensors, roots are the output tensors. By tracing this graph from roots to leaves,
# you can automatically compute the gradients using the chain rule.


a=torch.randn(2,2)
b=torch.randn(2,2)
c=torch.randn((2,2),requires_grad=True)

print("a+b requires grad: ",(a+b).requires_grad)
print("a+c requires grad: ",(a+c).requires_grad)

print("############################# frozen parameters #############################")
# In a NN, parameters that don’t compute gradients are usually called frozen parameters.
# Two reason two use froze parameters:
#1) Performance: if you know in advance that you won’t need the gradients of those parameters
#2) Finetuning a pretrained network