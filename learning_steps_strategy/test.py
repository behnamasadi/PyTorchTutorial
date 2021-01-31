import os
import torch
import torchvision

import matplotlib.pyplot as plt
import numpy as np

from net import *

os.system('python --version')
print('pytorch version:', torch.__version__)

if(torch.cuda.is_available()):
    print('cuda is available')
    print('cuda device count',torch.cuda.device_count())

    print('current device is:',torch.cuda.current_device())

    print('device name',torch.cuda.get_device_name(0))

    print('nvcc version: ')
    os.system('nvcc --version')
    print('nvidia-smi:')
    os.system('nvidia-smi')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('device is: ', device)

r_mean, g_mean, b_mean, r_std, b_std, g_std=0.49139968, 0.48215841, 0.44653091, 0.24703223, 0.24348513, 0.26158784

# train_transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
#                                                 torchvision.transforms.Normalize(
#                                                     (r_mean, g_mean, b_mean), (r_std, b_std, g_std)  ) ])

test_transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

CIFAR10_test_dataset=torchvision.datasets.CIFAR10(root='../data',download=True,transform=test_transform,train=False)

testloader = torch.utils.data.DataLoader(CIFAR10_test_dataset, batch_size=4,
                                         shuffle=False, num_workers=0)


PATH = './cifar_net.pth'

net = Net()
net.load_state_dict(torch.load(PATH))


correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        #print("labels:",labels.shape)

        #print("images:",images.shape)
        outputs = net(images)
        #print('outputs:',outputs.shape)
        _, predicted = torch.max(outputs.data, 1)
        print('predicted:',predicted)
        #print("labels:", labels)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))