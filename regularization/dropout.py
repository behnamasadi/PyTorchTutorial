import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np
import wandb

# https://docs.wandb.ai/guides/integrations/pytorch
# https://colab.research.google.com/drive/1uX4QhQ4levE1FMvFRlqzvYOeK-MBqtcu?usp=sharing#scrollTo=IzqTsmA_OfJU


device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))


BATCH_SIZE = 32

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='../data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='../data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=2)

CLASS_NAMES = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def show_batch(image_batch, label_batch):
    plt.figure(figsize=(10,10))
    for n in range(25):
        ax = plt.subplot(5,5,n+1)
        img = image_batch[n] / 2 + 0.5     # unnormalize
        img = img.numpy()
        plt.imshow(np.transpose(img, (1, 2, 0)))
        plt.title(CLASS_NAMES[label_batch[n]])
        plt.axis('off')


def train(model, device, train_loader, optimizer, criterion, epoch, steps_per_epoch=20):
    # Switch model to training mode. This is necessary for layers like dropout, batchnorm etc which behave differently in training and evaluation mode
    model.train()

    train_loss = 0
    train_total = 0
    train_correct = 0

    # We loop over the data iterator, and feed the inputs to the network and adjust the weights.
    for batch_idx, (data, target) in enumerate(train_loader, start=0):
        # Load the input features and labels from the training dataset
        data, target = data.to(device), target.to(device)

        # Reset the gradients to 0 for all learnable weight parameters
        optimizer.zero_grad()

        # Forward pass: Pass image data from training dataset, make predictions about class image belongs to (0-9 in this case)
        output = model(data)

        # Define our loss function, and compute the loss
        loss = criterion(output, target)
        train_loss += loss.item()

        scores, predictions = torch.max(output.data, 1)
        train_total += target.size(0)
        train_correct += int(sum(predictions == target))

        # Reset the gradients to 0 for all learnable weight parameters
        optimizer.zero_grad()

        # Backward pass: compute the gradients of the loss w.r.t. the model's parameters
        loss.backward()

        # Update the neural network weights
        optimizer.step()

    acc = round((train_correct / train_total) * 100, 2)
    print('Epoch [{}], Loss: {}, Accuracy: {}'.format(epoch, train_loss / train_total, acc), end='')
    wandb.log({'Train Loss': train_loss / train_total, 'Train Accuracy': acc, 'Epoch': epoch})


def test(model, device, test_loader, criterion, classes):
    # Switch model to evaluation mode. This is necessary for layers like dropout, batchnorm etc which behave differently in training and evaluation mode
    model.eval()

    test_loss = 0
    test_total = 0
    test_correct = 0

    example_images = []
    with torch.no_grad():
        for data, target in test_loader:
            # Load the input features and labels from the test dataset
            data, target = data.to(device), target.to(device)

            # Make predictions: Pass image data from test dataset, make predictions about class image belongs to (0-9 in this case)
            output = model(data)

            # Compute the loss sum up batch loss
            test_loss += criterion(output, target).item()

            scores, predictions = torch.max(output.data, 1)
            test_total += target.size(0)
            test_correct += int(sum(predictions == target))

    acc = round((test_correct / test_total) * 100, 2)
    print(' Test_loss: {}, Test_accuracy: {}'.format(test_loss / test_total, acc))
    wandb.log({'Test Loss': test_loss / test_total, 'Test Accuracy': acc})


class NetUnregularized(nn.Module):
    def __init__(self, input_shape=(3, 32, 32)):
        super(NetUnregularized, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)

        self.pool = nn.MaxPool2d(2, 2)

        n_size = self._get_conv_output(input_shape)

        self.fc1 = nn.Linear(n_size, 512)
        self.fc2 = nn.Linear(512, 10)

    def _get_conv_output(self, shape):
        batch_size = 1
        input = torch.autograd.Variable(torch.rand(batch_size, *shape))
        output_feat = self._forward_features(input)
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size

    def _forward_features(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class NetRegularized(nn.Module):
    def __init__(self, input_shape=(3, 32, 32)):
        super(NetRegularized, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)

        self.pool = nn.MaxPool2d(2, 2)

        n_size = self._get_conv_output(input_shape)

        self.fc1 = nn.Linear(n_size, 512)
        self.fc2 = nn.Linear(512, 10)

        self.dropout = nn.Dropout(0.25)

    def _get_conv_output(self, shape):
        batch_size = 1
        input = torch.autograd.Variable(torch.rand(batch_size, *shape))
        output_feat = self._forward_features(input)
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size

    def _forward_features(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


#net = NetUnregularized().to(device)
net = NetRegularized().to(device)
print(net)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())


wandb.init(project='regularization', entity='behnamasadi')
wandb.watch(net, log='all')

# # 2. Save model inputs and hyperparameters
# config = wandb.config
# config.learning_rate = 0.01

# 3. Track metrics
# wandb.log({'accuracy': train_acc, 'loss': train_loss})

# 4. Track hyperparameters
# wandb.config.dropout = 0.2

for epoch in range(15):
    train(net, device, trainloader, optimizer, criterion, epoch)
    test(net, device, testloader, criterion, CLASS_NAMES)

print('Finished Training')
wandb.finish()