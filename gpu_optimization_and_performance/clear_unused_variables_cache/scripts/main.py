import torch
import time

x = torch.randn(20000, 20000, device='cuda')  # ~1.6 GB VRAM
y = x  # another reference


del x
print("del x")

time.sleep(5)  # Sleep for 10 seconds


del y  # refcount is now 0 → GPU memory released
print("del y")

torch.cuda.empty_cache()


time.sleep(5)  # Sleep for 10 seconds
