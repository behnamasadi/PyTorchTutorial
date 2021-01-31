#https://github.com/vdumoulin/conv_arithmetic

import sys
sys.path.append("../utility")
import shape_calculator
import torch



kernel_size=(5,5)
stride=(1,1)
padding=(0,0)
dilation=(1,1)


print("########################### Conv2d ###########################")
# in_channels
# So the in_channels in the beginning is 3 for images with 3 channels (colored images).
# For images black and white it should be 1.
# Some satellite images should have 4.


# out_channels
# The out_channels is what convolution will produce so these are the number of filters.

m = torch.nn.Conv2d(in_channels=1,
                    out_channels=1,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    padding_mode='zeros')

# batch x channel x row x column

channels=1
number_of_batch=20
height=28
width=28
input = torch.randn(number_of_batch, channels, height, width)
print("input shape is:",input.shape)
x=m(input)
print("output shape is:",x.shape)
print("calculated height and width:",shape_calculator.conv2d(height,width,padding,dilation,kernel_size,stride))


print("########################### displaying an image ###########################")

import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

pre_process = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor(),
     transforms.Normalize(mean=[0.5], std=[0.5])])

# original colors
#dataFromFolders = datasets.ImageFolder(root="../data/images",transform=transforms.ToTensor())

# single channel
dataFromFolders = datasets.ImageFolder(root="../data/images",transform=pre_process)

folderloader = DataLoader(dataFromFolders, batch_size=4,shuffle=True)
images,labels = iter(folderloader).next()

print("type(images[0]):", type(images[0]))
np_image=images[0]
np_image=images[0].permute(1,2,0)
plt.imshow(np_image,cmap='Greys')
plt.show()

print("########################### applying a conv2d on an image ###########################")

horizontal_edges = torch.tensor([[-1,0,1],[-1,0,1],[-1,0,1]],dtype=torch.float32)
vertical_edges=torch.transpose(horizontal_edges,0, 1)
horizontal_edges = horizontal_edges.reshape(1,1,3,3)

print("filter (horizontal edges):",horizontal_edges)
print("filter.shape:", horizontal_edges.shape)

output_image=torch.nn.functional.conv2d(input =images,weight=horizontal_edges)
print("images.shape:",images.shape)

print("output_image.shape:",output_image.shape)

plt.imshow(output_image[0].permute(1,2,0),cmap='Greys')
plt.show()

vertical_edges = vertical_edges.reshape(1,1,3,3)
output_image=torch.nn.functional.conv2d(input =images,weight=vertical_edges)
plt.imshow(output_image[0].permute(1,2,0),cmap='Greys')
plt.show()