# lecture 6 cs231n 2017 1:20:19
# 1) Pre process data:
# Normalize the data (Z-Score Normalization)
# Turn the data into zero mean, and std=1
# you should know mean and std of every channel in advance, this could be done only if you iterate over all images
# i.e. CIFAR10
# r_mean, g_mean, b_mean, r_std, b_std, g_std=0.49139968, 0.48215841, 0.44653091, 0.24703223, 0.24348513, 0.26158784
#
# transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
#                                                 torchvision.transforms.Normalize(
#                                                     (r_mean, g_mean, b_mean),
#                                                     (r_std, b_std, g_std)  ) ])


# 2) Choose architecture
# 2-1) Weight initialization
# I) xavier
# II) kaiming-he
# 2-2) Check if the loss is reasonable
# I) Disable regularization, do a forward and check the loss
# II) Crank up the regularization and the loss should goes up,
# III) Feed a small portion of data, turn off regularization, it should easily overfit
# and loss goes to zero, train accuracy =1

# batch normalization
# drop out



# plotting

# don’t use autograd of not necessary (use with torch.no_grad() if possible)
# only push tensors to GPU, if they are actually needed
# try to avoid loops over tensor dimensions (slows things down)
# try to free graphs as soon as possible (use detach or item whenever you can) to avoid memory leaks