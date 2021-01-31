import os
import torch
import torchvision

import matplotlib.pyplot as plt
import numpy as np

from  net import *

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
#                                                     (r_mean, g_mean, b_mean),
#                                                     (r_std, b_std, g_std)  ) ])


train_transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


# loading data and prepeocessing
CIFAR10_train_dataset=torchvision.datasets.CIFAR10(root='../data',download=True,transform=train_transform,train=True)

trainloader = torch.utils.data.DataLoader(CIFAR10_train_dataset, batch_size=4,
                                          shuffle=True, num_workers=0)



def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    #img = np.array([r_mean, g_mean, b_mean]) +img / np.array([r_std, b_std, g_std])
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()
# show images
#imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % CIFAR10_train_dataset.classes[labels[j]] for j in range(4)))






net=Net()
loss_function=torch.nn.CrossEntropyLoss()
#optimizer=torch.optim.Adam(net.parameters(),betas=(0.9, 0.999), eps=1e-08)
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


epochs=2
running_loss=0.0

for epoch in range(epochs):
    for i, data in enumerate(trainloader,0):
        images, classes=data
        optimizer.zero_grad()
        outputs =net(images)
        loss=loss_function(outputs,labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0


PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)