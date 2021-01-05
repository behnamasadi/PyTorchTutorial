# provides minibatching, shuffling, and multithreading

import torch


from torch.utils.data import  TensorDataset, DataLoader

N, D_in, H, D_out=64,1000,100,10
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

loader = DataLoader(TensorDataset(x, y), batch_size = 8)
for x_batch, y_batch in loader:
    pass

print("###################################### datasets loader ######################################")
import torchvision.transforms as transformers
import torchvision

# The files are read and converted into PIL images

# full list of available datasets: 
# https://subscription.packtpub.com/book/big_data_and_business_intelligence/9781789534092/1/ch01lvl1sec13/loading-data
trainset_CIFAR10=torchvision.datasets.CIFAR10(root="../data",download=True)

# you can add an other parameter to the constructor and transform them into tensor ,transform=transformers.ToTensor()
# trainset_CIFAR10=torchvision.datasets.CIFAR10(root="../data",download=True, transform=transformers.ToTensor())

print("trainset_CIFAR10 is a list of size:",len(trainset_CIFAR10))
print("every item in the list is a:",type(trainset_CIFAR10[0]))
print("image type is:",type(trainset_CIFAR10[0][0]))
print("label type is:",type(trainset_CIFAR10[0][1]))
print("label :",trainset_CIFAR10[0][1])
# display the PIL image
trainset_CIFAR10[0][0].show(title=None, command=None)

print("###################################### display tensor images ######################################")
import matplotlib.pyplot  as plt
trainset_MNIST= torchvision.datasets.MNIST(root="../data",download=True,transform=transformers.ToTensor())
np_image=trainset_MNIST[0][0].permute(1,2,0)
plt.imshow(np_image)
plt.show()


print("###################################### DataLoader ######################################")
# to shuffle data and load them in batch we can use DataLoader
trainset_CelebA=torchvision.datasets.MNIST(root="../data",download=True, transform=transformers.ToTensor())
testset_CelebA=torchvision.datasets.MNIST(root="../data",download=True, transform=transformers.ToTensor())
batch_size=10
trainset_loader=torch.utils.data.DataLoader(trainset_CelebA,batch_size=batch_size,shuffle=True,num_workers=4)
data_iter=iter(trainset_loader)

# calling next() will giveus a batch
images, labels=data_iter.next()
print("The batch size: {0} is and labels in this batch are:".format(batch_size),labels)
print("image sizes in this batch are: {}".format(images[0].size()))

print("###################################### Custom Dataset ######################################")
# you can inherit from Dataset class and make your dataset. pleasechech custome_dataset.py


print("###################################### ImageFolder ######################################")
my_trainingset=torchvision.datasets.ImageFolder(root="../data/images",transform=transformers.ToTensor())
my_trainingset_loader=torch.utils.data.DataLoader(my_trainingset,batch_size=batch_size,shuffle=True,num_workers=4)
data_iter=iter(my_trainingset_loader)
images, labels=data_iter.next()
print("Notice that the retrieved labels using DataLoader are represented by integers",labels)

print('###################################### Transform ######################################')
# We define some transforms because images are of different sizes and shape  etc

# Normalizing helps to keep the Network weights near zero which in turn makes back propagation more stable.
# Without normalization, the network will fail to learn properly.
# Normalize does the following for each channel:
# input[channel] = (input[channel] - mean[channel]) / std[channel]
# This will normalize the image in the range [-1,1].
# the minimum value 0 will be converted to (0-0.5)/0.5=-1, the maximum value of 1 will be converted to (1-0.5)/0.5=1.

r_mean, g_mean, b_mean, r_std, b_std, g_std=0.5, 0.5, 0.5, 0.5, 0.5, 0.5
transformers.Normalize((r_mean,g_mean,b_mean), (r_std,g_std,b_std))

#  your input image is resized to be of size (256, 256)
transformers.Resize(256)

#  Crops the center part of the image of shape (224, 224)
transformers.CenterCrop(224)

# This will extract a patch of size (224, 224) from your input image randomly.
transformers.RandomResizedCrop(224)

# this will
transformers.Compose([transformers.CenterCrop(224),transformers.RandomResizedCrop(224)])
