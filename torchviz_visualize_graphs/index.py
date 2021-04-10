import torchviz
import torch

import torchvision
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader



#https://github.com/szagoruyko/pytorchviz/blob/master/examples.ipynb

# Example 1
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = torch.nn.Linear(in_features=4,out_features= 2, bias=True)
        return
    def forward(self, input):
        return torch.tanh(self.l1(input))

model=Net()
input=torch.randn(size=[1,4])
print(model(input))


dot=torchviz.make_dot(model(input),params=dict(model.named_parameters()),show_attrs=True, show_saved=True )
dot.format='svg'
dot.render(filename='example_graph', directory='../images')




# Example 2

a=torch.tensor([1.2],requires_grad=True)
b=torch.tensor([2.2],requires_grad=True)
c=torch.tensor([0.2],requires_grad=True)
d=torch.tensor([0.8],requires_grad=True)
e=torch.tensor([7.],requires_grad=True)

f=d*((a+b)*(c))+e

f.backward()

f_params={'a':a,'b':b,'c':c,'d':d,'e':e}


dot=torchviz.make_dot(f,params=f_params,show_attrs=True, show_saved=True )
dot.format='svg'
dot.render(filename='simple_graph', directory='../../images')



# Example 3


train_loader=DataLoader(MNIST(root='../data/MNIST', train=True, download=True)
                        ,batch_size=100,
                        num_workers=8,
                        shuffle=True)

model=torchvision.models.resnet101(pretrained=True)
x = torch.zeros(1, 3, 224, 224, dtype=torch.float, requires_grad=False)
out = model(x)
torchviz.make_dot(out).render("../images/resnet101", format="svg")
