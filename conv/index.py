#https://github.com/vdumoulin/conv_arithmetic

import sys
sys.path.append("../utility")
import shape_calculator
import torch

kernel_size=(5,5)
stride=(1,1)
padding=(0,0)
dilation=(1,1)

m = torch.nn.Conv2d(in_channels=1,
                    out_channels=1,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    padding_mode='zeros')

channels=1
number_of_batch=20
height=28
width=28
input = torch.randn(number_of_batch, channels, height, width)
print("input shape is:",input.shape)
x=m(input)
print("output shape is:",x.shape)
print("calculated height and width:",shape_calculator.shape_calculator(height,width,padding,dilation,kernel_size,stride))




import torch.nn.functional as F
from torch.utils.data import  TensorDataset, DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

pre_process = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor(),
     transforms.Normalize(mean=[0.5], std=[0.5])])
dataFromFolders = datasets.ImageFolder(root="../data/images",transform=pre_process)
folderloader = DataLoader(dataFromFolders, batch_size=4,shuffle=True)
images,labels = iter(folderloader).next()

print("", type(images[0]))
np_image=images[0]
#np_image=images[0].permute(1,2,0)
#plt.imshow(np_image)
#plt.show()



# stride: Union[_int, _size]=1,
# padding: Union[_int, _size]=0,
# dilation: Union[_int, _size]=1,

#dataloader.

#F.conv2d(input=img)
# nn.Conv2d
# torch.nn.functional.conv2d



# print(m)
# m.weight[:] =torch.tensor([[-1,0,1],[-1,0,1],[-1,0,1]])
# print(m.weight)
# print(m.weight.requires_grad)
# print(m.weight.data.requires_grad)
#
# #np_image=m(np_image)
# np_image = m(np_image[None, ...])
# plt.imshow(np_image)
# plt.show()



