import torch
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


a = torch.tensor(data=[2., 3.], requires_grad=True)
b = torch.tensor([6., 4.], requires_grad=True)
q = 3*a**3-b**2


external_grad = torch.tensor([1., 1.])
q.backward(gradient=external_grad)
print(a.grad)
print(9*a**2)
