# don’t use autograd of not necessary (use with torch.no_grad() if possible)
# only push tensors to GPU, if they are actually needed
# try to avoid loops over tensor dimensions (slows things down)
# try to free graphs as soon as possible (use detach or item whenever you can) to avoid memory leaks