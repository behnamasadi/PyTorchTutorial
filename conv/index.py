#https://github.com/vdumoulin/conv_arithmetic

import torch
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

m = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3,3), stride=1, padding=0, dilation=1,
                     padding_mode='zeros')
print(m)
m.weight[:] =torch.tensor([[-1,0,1],[-1,0,1],[-1,0,1]])
print(m.weight)
print(m.weight.requires_grad)
print(m.weight.data.requires_grad)

#np_image=m(np_image)
np_image = m(np_image[None, ...])
plt.imshow(np_image)
plt.show()



