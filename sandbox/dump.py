import torch
import torch.nn as nn
import yaml


# import os


# model = nn.Sequential(nn.Flatten(), nn.Linear(2, 10),
#                       nn.ReLU(),    nn.Linear(10, 1))


# if torch.cuda.is_available():
#     device = torch.device("cuda")
#     torch.cuda.manual_seed(42)
# else:
#     device = torch.device("cpu")
#     torch.manual_seed(42)


# model.to(device)

# if os.path.isfile("model.pth"):
#     print("Loading model from model.pth")
#     model.load_state_dict(torch.load("model.pth", weights_only=True))
# else:
#     print("Creating new model")


# N, C, H, W = 5, 1, 2, 1
# input = torch.randn(N, C, H, W, device=device)


# input = input.view(N, -1)
# print(model(input))


# torch.save(model.state_dict(), "model.pth")


with open("model_config.yaml", "r") as f:
    config = yaml.safe_load(f)


conv_layers = config['conv_layers']
print(conv_layers[0]["out_channels"])
