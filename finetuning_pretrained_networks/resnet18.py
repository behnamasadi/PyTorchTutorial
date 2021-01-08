import torch
import torchvision


model=torchvision.models.resnet18(pretrained=True)
for params in model.parameters():
    params.requiers_gard=False
print(type(model.fc))

print(model.fc.in_features)
print(model.fc.out_features)

model.fc=torch.nn.Linear(512,10)


optimizer=torch.optim.SGD(model.fc.parameters(),lr=1e-2,momentum=0.9)


# TORCH_MODEL_ZOO
# print(torch.utils.model_zoo.load_url())