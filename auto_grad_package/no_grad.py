import torch

x=torch.randn([2,3], requires_grad=True)
print(x)

# torch.no_grad() in this context manager in the __enter__() method, set_grad_enabled(False)
# so for tensor object requires_grad will turn into False

with torch.no_grad():
    y=2*x

print(y.requires_grad)