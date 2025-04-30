import torch
import torch.nn as nn
import yaml
import os
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# # Set KaggleHub cache directory
# os.environ['KAGGLE_HUB_CACHE_DIR'] = './data/kagglehub/'

# # import os


# # model = nn.Sequential(nn.Flatten(), nn.Linear(2, 10),
# #                       nn.ReLU(),    nn.Linear(10, 1))


# # if torch.cuda.is_available():
# #     device = torch.device("cuda")
# #     torch.cuda.manual_seed(42)
# # else:
# #     device = torch.device("cpu")
# #     torch.manual_seed(42)


# # model.to(device)

# # if os.path.isfile("model.pth"):
# #     print("Loading model from model.pth")
# #     model.load_state_dict(torch.load("model.pth", weights_only=True))
# # else:
# #     print("Creating new model")


# # N, C, H, W = 5, 1, 2, 1
# # input = torch.randn(N, C, H, W, device=device)


# # input = input.view(N, -1)
# # print(model(input))


# # torch.save(model.state_dict(), "model.pth")


# with open("model_config.yaml", "r") as f:
#     config = yaml.safe_load(f)


# conv_layers = config['conv_layers']
# print(conv_layers[0]["out_channels"])


CIFAR10_train_dataset = datasets.CIFAR10(root='../data', train=True,
                                         download=True, transform=transforms.ToTensor())


def calculate_mean_std(dataset):

    num_workers = min(4, os.cpu_count())
    CIFAR10_train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=64, shuffle=False, num_workers=num_workers)

    mean = 0.0
    std = 0.0
    num_images = 0
    for batch_images, batch_labels in CIFAR10_train_dataloader:
        # batch_images is a tensor of shape (B, C, H, W)
        batch_images = batch_images.view(
            batch_images.size(0), batch_images.size(1), -1)

        # batch_images.mean(2) is a tensor of shape (B, C) indicating the mean of each channel
        # batch_images.mean(2).sum(0) is a tensor of shape (C) indicating the mean of each channel, so we have mean of each channel for all images in the batch
        mean += batch_images.mean(2).sum(0)
        std += batch_images.std(2).sum(0)
        num_images += batch_images.size(0)

    return mean / num_images, std / num_images


# print(CIFAR10_train_dataset.data.shape)
# img, label = CIFAR10_train_dataset[0]
# print(img.shape)
# print(label)

mean, std = calculate_mean_std(CIFAR10_train_dataset)
print(mean, std)
exit()

# # print(dataset[0][0].shape)
# print(CIFAR10_train_dataset.data.shape)
# # print(CIFAR10_train_dataset[0][0].shape)
# print(CIFAR10_train_dataset[0][0].mean())
# print(CIFAR10_train_dataset[0][0].std())
# print("CIFAR10_train_dataset.data.min(): ", CIFAR10_train_dataset.data.min())
# print("CIFAR10_train_dataset.data.max(): ", CIFAR10_train_dataset.data.max())
