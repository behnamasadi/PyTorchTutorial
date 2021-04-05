import torch
import torchvision

# Normalization (min-max Normalization) [0,1]
# Standardization (mu=0, std=1)
# in pytorch normalization means: mu=0, std=1

print('\n############################ CIFAR10 dataset #######################################\n')

print('\nIf you read the data directly from pytorch, they are in range of [0,255]\n')

train_transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()])


CIFAR10_train_dataset=torchvision.datasets.CIFAR10(root='../data',download=True,transform=train_transform,train=True)

min_value=CIFAR10_train_dataset.data.min()
max_value=CIFAR10_train_dataset.data.max()

print("CIFAR10_train_dataset.data.min(): ",min_value)
print("CIFAR10_train_dataset.data.max(): ",max_value)


r_mean, g_mean, b_mean=CIFAR10_train_dataset.data.mean(axis=(0,1,2))
r_std, g_std, b_std=CIFAR10_train_dataset.data.std(axis=(0,1,2))

print("mean of r, g, b channel:",r_mean, g_mean, b_mean)
print("standard deviation  of r, g, b channel:",r_std, g_std, b_std)


print('############################################################################')

print('\nIf you load the data with DataLoader without using any transformer they will be in the range of [0,1]\n')



trainloader = torch.utils.data.DataLoader(CIFAR10_train_dataset, batch_size=4,
                                          shuffle=True, num_workers=2)



dataiter = iter(trainloader)
images, labels = dataiter.next()
print("images.min(): ",images.min())
print("images.max(): ", images.max())

print('############################################################################')

print('\nSince you want to load the input to your network in the form of normal distribution with mean 0 and std 1'
      'you should compute the mean and std of your data in advance from dataset directly '
      ' divide it by max value (since DataLoader will make it in the range of [0,1] ) and use it when loading data'
      ' from DataLoader \n')



r_mean, g_mean, b_mean=[r_mean/max_value,  g_mean/max_value, b_mean/max_value]
r_std, std_g, b_std=[r_std/max_value, g_std/max_value, b_std/max_value]

train_transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize(
                                                    (r_mean, g_mean, b_mean),
                                                    (r_std, b_std, g_std)  ) ])


CIFAR10_train_dataset=torchvision.datasets.CIFAR10(root='../data',download=True,transform=train_transform,train=True)

trainloader = torch.utils.data.DataLoader(CIFAR10_train_dataset, batch_size=4,
                                          shuffle=True, num_workers=2)



dataiter = iter(trainloader)
images, labels = dataiter.next()
print('\nNow data are in the form of normal distribution\n')

print("images.min(): ",images.min())
print("images.max(): ", images.max())
print("shape of batch: batch_size x channel x row x column: ",images.shape)
print("shape of training dataset: ",CIFAR10_train_dataset.data.shape)
print("size of images: row x column x channel: ",CIFAR10_train_dataset.data[0].shape)


print('\n############################ MNIST dataset #######################################\n')

print('\n loading data using Normalize((0.1307,), (0.3081,) \n')

batch_size_train = 64
batch_size_test = 1000

transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                          torchvision.transforms.Normalize((0.1307,), (0.3081,)) ])

MNIST_train_dataset = torchvision.datasets.MNIST('../data', train=True, download=True,transform=transform)

MNIST_trainloader = torch.utils.data.DataLoader(MNIST_train_dataset,batch_size=batch_size_train, shuffle=True)
dataiter = iter(MNIST_trainloader)

print('after standardization, the values are in form of normal distribution:')
images, labels = dataiter.next()
print("images.min(): ",images.min())
print("images.max(): ", images.max())
print("shape of batch: batch_size x channel x row x column: ",images.shape)