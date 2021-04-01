#https://github.com/szagoruyko/pytorchviz/blob/master/examples.ipynb
import torchviz
import torch

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

dot=torchviz.make_dot(model(input),params=dict(model.named_parameters()) )
dot.format='svg'
dot.render(filename='example_graph', directory='../images')
#dot.save('images/graph.svg')