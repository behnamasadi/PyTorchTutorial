# Make Sure That Pytorch Using GPU To Compute
import torch
import os



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

#dtype = torch.cuda.FloatTensor

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)



# generally speaking, the pattern is:
# use .cuda() on any input batches/tensors
# use .cuda() on your network module, which will hold your network, like:

class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()


model = MyModel()
model.cuda()


# Datatype                     dtype                   CPU tensor                   GPU tensor
#32-bit floating point     torch.float32           torch.FloatTensor             torch.cuda.FloatTensor
#                          or torch.float

x=torch.randn([2,3],dtype=torch.float32)
print(type(x))
print(type(x.data))
print(x.dtype) # returns dtype
print(x.type()) # CPU tensor or GPU tensor
print(torch.Tensor(1).dtype)
