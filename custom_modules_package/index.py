import torch
import torch.nn.functional as F
from torchvision.datasets import MNIST

class MyConvNet(torch.nn.Module):
    def __init__(self):
        self.conv1=torch.nn.Conv2d(1,32,kernel_size=5)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3)
        self.fc1=torch.nn.Linear(1600,100)
        self.fc2 = torch.nn.Linear(100,10)

    def forward(self,x):
        x=self.conv1(x)
        x=F.max_pool2d(x,kernel_size=2,stride=2)
        x=F.relu(x)
        x=F.relu(F.max_pool2d(self.conv2(x), kernel_size=2, stride=2))
        x=x.view(-1,1600)
        x=F.relu(self.fc1(x))
        x=self.fc2(x)
        return x

files_path='../data/'
mnist=MNIST(files_path, train=True,download=True)
d=mnist.data.type('torch.DoubleTensor')
d=d.view(d.size(0),1,28,28)


model=MyConvNet().double()
y = model(d[0:1, :, :, :])

# https://discuss.pytorch.org/t/why-model-to-device-wouldnt-put-tensors-on-a-custom-layer-to-the-same-device/17964/3

# generally speaking, the pattern is:
# use .cuda() on any input batches/tensors
# use .cuda() on your network module, which will hold your network, like:

class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()


model = MyModel()
model.cuda()

# training=self.training

# https://discuss.pytorch.org/t/model-train-and-model-eval-vs-model-and-model-eval/5744
