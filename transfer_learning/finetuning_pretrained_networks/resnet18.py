import torch
import torchvision
import torchviz


resnet18=torchvision.models.resnet18(pretrained=True)
for params in resnet18.parameters():
    params.requiers_gard=False

# How to find model input size

print("resnet18 input size: ", resnet18.fc.in_features)
print("resnet18 output size: ",resnet18.fc.out_features)


#  resnet18 has an averagepool layer at the end.
#  So the input size does not matter much provided the feature map size is greater than kernel size.

input=torch.randn(size=[1,3,128,128])

resnet18_graph=torchviz.make_dot(resnet18(input) ,dict(resnet18.named_parameters()))
resnet18_graph.format='svg'
resnet18_graph.save('images/resnet18_graph')
resnet18_graph.render()

# Let’s say we want to finetune the model on a new dataset with 10 labels. In resnet,
# the classifier is the last linear layer model.fc. We can simply replace it with a new linear layer
# (unfrozen by default) that acts as our classifier.

resnet18.fc=torch.nn.Linear(512,10)

# Now all parameters in the model, except the parameters of model.fc, are frozen.
# The only parameters that compute gradients are the weights and bias of model.fc.

optimizer=torch.optim.SGD(resnet18.fc.parameters(),lr=1e-2,momentum=0.9)


# TORCH_resnet18_ZOO
# print(torch.utils.resnet18_zoo.load_url())