import torch
import torchvision
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchviz import make_dot


train_loader=DataLoader(MNIST(root='./data/MNIST', train=True, download=True)
                        ,batch_size=100,
                        num_workers=8,
                        shuffle=True)

model=torchvision.models.resnet101(pretrained=True)
x = torch.zeros(1, 3, 224, 224, dtype=torch.float, requires_grad=False)
out = model(x)
make_dot(out).render("resnet101", format="svg")
#make_dot(model2(x), params=dict(model2.named_parameters())).render("model2", format="svg")



