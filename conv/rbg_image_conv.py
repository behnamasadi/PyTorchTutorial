import torch
import torchvision

# awesome visualization:
# http://datahacker.rs/convolution-rgb-image/


batch_size=10
n_channel=3
n_row=27
n_column=27

images=torch.randn(batch_size,n_channel,n_row,n_column,requires_grad=False)
# print(images[0])

r_edges=[[-1,0,1],[-1,0,1],[-1,0,1]]
b_edges=[[-1,0,1],[-1,0,1],[-1,0,1]]
g_edges=[[-1,0,1],[-1,0,1],[-1,0,1]]

horizontal_edges = torch.tensor( [r_edges,r_edges,r_edges] ,dtype=torch.float32)
print("filter.shape:", horizontal_edges.shape)

horizontal_edges = horizontal_edges.reshape(1,3,3,3)

print("filter.shape:", horizontal_edges.shape)

output_image=torch.nn.functional.conv2d(input =images, weight=horizontal_edges)

print(output_image.shape)

