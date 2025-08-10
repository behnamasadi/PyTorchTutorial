import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torchvision

################################ Setting log directory, clearning previous run ################################


logdir='../runs/tensorboard_demo'

import shutil
import os
if os.path.exists(logdir):
    shutil.rmtree(logdir, ignore_errors=True)
    print(logdir+ " cleared")

os.mkdir(logdir)


################################ Running tensorboard  ################################

#!/usr/bin/env python
import os
import signal
import subprocess

proc = subprocess.Popen("tensorboard --logdir="+logdir, shell=True, preexec_fn=os.setsid)



# $ tensorboard --logdir=runs --port=6006
# $ tensorboard --logdir=runs --host <public_ip> --port=6006

################################ Add scalar and scalars ################################
# scalar : It will plot just one graph
# scalars : It will plot multiple graphs at once

writer=SummaryWriter(log_dir=logdir)
for step in np.arange(-360,360):
    writer.add_scalar('sin', np.sin(step*(np.pi/180)),step)
    writer.add_scalars('sin and cos', { 'sin' : np.sin(step*(np.pi/180)) ,  'cos' : np.cos(step*(np.pi/180))},step)
writer.close()


################################ Add image and images ################################
pre_procc=torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
mnits_data=torchvision.datasets.MNIST(root='../data', download=True, transform=pre_procc, train=True )

batch_size=4
mnist_data_loader=torch.utils.data.DataLoader(dataset=mnits_data ,batch_size=batch_size, shuffle=True)

mnist_data_loader_iter= iter(mnist_data_loader)
images,lables=next(mnist_data_loader_iter)
print(lables[0].item())
writer.add_image(str(lables[0].item()),images[0])
writer.close()


images,lables=next(mnist_data_loader_iter)
print(lables.shape)
writer.add_images('MNIST group'+str(lables.numpy()), images)
writer.close()


################################ Add histogram ################################
sigma=1
mean=0
np.random.normal(loc=mean, scale=sigma, size=100)
num_steps=5
for step in np.arange(num_steps):
    # writer.add_histogram('sigma is'+str((step+1)*sigma),np.random.normal(loc=mean, scale=(step+1)*sigma, size=1000),step)
    writer.add_histogram('weights',np.random.normal(loc=mean, scale=(step + 1) * sigma, size=2000), step)
writer.close()
################################ Add model ################################
import torch.nn.functional as F

class MyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)

        self.fc1 = torch.nn.Linear(in_features=12*4*4, out_features=120)
        self.fc2 = torch.nn.Linear(in_features=120, out_features=60)
        self.out = torch.nn.Linear(in_features=60, out_features=10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size = 2, stride = 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size = 2, stride = 2)
        x = torch.flatten(x,start_dim = 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)

        return x

train_set = torchvision.datasets.FashionMNIST(root="../data",train = True, download=True,transform=torchvision.transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(train_set,batch_size = 100, shuffle = True)




model = MyNet()
images, labels = next(iter(train_loader))
grid = torchvision.utils.make_grid(images)
writer.add_image("MNIST Fashion images", grid)
writer.add_graph(model, images)
writer.close()



################################ Precision Recall Curve ################################

# https://pytorch.org/docs/stable/tensorboard.html
# https://towardsdatascience.com/a-complete-guide-to-using-tensorboard-with-pytorch-53cb2301e8c3

input("Press Enter to continue...")
os.killpg(proc.pid, signal.SIGTERM)