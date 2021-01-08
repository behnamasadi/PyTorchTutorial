import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torchvision

logdir='runs'

import shutil
import os
if os.path.exists(logdir):
    shutil.rmtree(logdir, ignore_errors=True)
    print(logdir+ " cleared")

os.mkdir(logdir)


#!/usr/bin/env python
import os
import signal
import subprocess

proc = subprocess.Popen("tensorboard --logdir=runs", shell=True, preexec_fn=os.setsid)



# $ tensorboard --logdir=runs --port=6006
# $ tensorboard --logdir=runs --host <public_ip> --port=6006

################################ Add scalar and scalars ################################
# scalar : It will plot just one graph
# scalars : It will plot multi graphs at once

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

class MyNet(torch.nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.conv1=torch.nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5,stride=1,dilation=1)
        self.fc1=torch.nn.Linear()

    def forward(self,x):
        pass





input("Press Enter to continue...")
os.killpg(proc.pid, signal.SIGTERM)