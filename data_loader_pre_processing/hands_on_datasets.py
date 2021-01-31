import torch
import torchvision

# https://www.kaggle.com/uvxy1234/cifar-10-implementation-with-pytorch
# https://github.com/kuangliu/pytorch-cifar/blob/master/main.py

# The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class.

train_transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()])


# loading data and prepeocessing
CIFAR10_train_dataset=torchvision.datasets.CIFAR10(root='../data',download=True,transform=train_transform,train=True)
CIFAR10_test_dataset=torchvision.datasets.CIFAR10(root='../data',download=True,transform=train_transform,train=False)

print("loading dataset:",type(CIFAR10_train_dataset))
print("mean of r, g, b channel:",CIFAR10_train_dataset.data.mean(axis=(0,1,2)))
print("standard deviation  of r, g, b channel:",CIFAR10_train_dataset.data.std(axis=(0,1,2)))


print('\n###################################################################\n')


print('The min and max values directly from dataset (i.e. CIFAR10_train_dataset.data[0]) are between 0 and 255')
print("CIFAR10_train_dataset.data.min(): ",CIFAR10_train_dataset.data.min())
print("CIFAR10_train_dataset.data.max(): ",CIFAR10_train_dataset.data.max())

print('\n###################################################################\n')

print("CIFAR10 classes are:",CIFAR10_train_dataset.classes)

print('\n###################################################################\n')

print("shape of taring dataset: ",CIFAR10_train_dataset.data.shape)
print("size of images: row x column x channel: ",CIFAR10_train_dataset.data[0].shape)


trainloader = torch.utils.data.DataLoader(CIFAR10_train_dataset, batch_size=4,
                                          shuffle=True, num_workers=2)


print('the min and max values from DataLoader are between 0 and 1 (assuming no normalization being done)')

dataiter = iter(trainloader)
images, labels = dataiter.next()
print("images.min(): ",images.min())
print("images.max(): ", images.max())

print('\n###################################################################\n')

print("shape of batch: batch_size x channel x row x column: ",images.shape)


#  we have calculated mean and std from CIFAR10_train_dataset.data.std(axis=(0,1,2)) and dividing by 255 since
# the output of DataLoader are between 0 and 1
print(CIFAR10_train_dataset.data.mean(axis=(0,1,2))/255)
print(CIFAR10_train_dataset.data.std(axis=(0,1,2))/255)
r_mean, g_mean, b_mean, r_std, b_std, g_std=0.49139968, 0.48215841, 0.44653091, 0.24703223, 0.24348513, 0.26158784

train_transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                 torchvision.transforms.Normalize( (r_mean, g_mean, b_mean), (r_std, b_std, g_std)  ) ])
CIFAR10_train_dataset=torchvision.datasets.CIFAR10(root='../data',download=True,transform=train_transform,train=True)

trainloader = torch.utils.data.DataLoader(CIFAR10_train_dataset, batch_size=4,
                                          shuffle=True, num_workers=2)

dataiter = iter(trainloader)
print('\n###################################################################\n')
print('after standardization, the values are in form of normal distribution:')
images, labels = dataiter.next()
print("images.min(): ",images.min())
print("images.max(): ", images.max())
print("shape of batch: batch_size x channel x row x column: ",images.shape)