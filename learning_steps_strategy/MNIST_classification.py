# Ref: https://nextjournal.com/gkoehler/pytorch-mnist

import torch
import torchvision
import torch.nn.functional as F


import matplotlib.pyplot as plt

def displayImages(training_data,training_targets):
    fig = plt.figure()
    for i in range(6):
        plt.subplot(2,3,i+1)
        plt.tight_layout()
        plt.imshow(training_data[i][0], cmap='gray', interpolation='none')
        plt.title("Ground Truth: {}".format(training_targets[i]))
        plt.xticks([])
        plt.yticks([])
    plt.show()





# For repeatable experiments we have to set random seeds for anything using random number generation
random_seed = 1
# cuDNN uses nondeterministic algorithms which can be disabled
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

n_epochs = 1
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10

# num_workers > 1 to use subprocesses to asynchronously load data or using pinned RAM (via pin_memory)
# to speed up RAM to GPU transfers.

num_workers=4




data_transformer=torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.1307,), (0.3081,))])


train_dataset=torchvision.datasets.MNIST(root='../data', train= True, transform = data_transformer, download = True)
test_dataset=torchvision.datasets.MNIST(root='../data', train= False, transform = data_transformer, download = True)


train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_train,
                                          shuffle=True, num_workers=num_workers)


test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size_test,
                                          shuffle=True, num_workers=num_workers)

train_iter=iter(train_loader)


example_data, example_targets=train_iter.next()
# Batch size of train, Number of Channels, Height, Width
print("Batch size of train, Number of Channels, Height, Width:", example_data.shape)
print(example_targets.data)

# "lables:",
displayImages(example_data, example_targets)


train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]



class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1=torch.nn.Conv2d(in_channels = 1,out_channels = 10,kernel_size=5)
        self.conv2=torch.nn.Conv2d(in_channels = 10,out_channels = 20,kernel_size=5)
        self.conv2_drop = torch.nn.Dropout2d()
        self.fc1 = torch.nn.Linear(320, 50)
        self.fc2 = torch.nn.Linear(50, 10)
    def forward(self,x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

    def reset_parameters(self):
        print("****************** reset parameters happend ****************** ")

    def initialize_weight(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_uniform_(m.weight)
            elif isinstance(m,torch.nn.BatchNorm2d):
                pass

def train(n_epochs,optimizer,network,criterion):
    network.train()

    for epoch in range(n_epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = network(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))
                train_losses.append(loss.item())
                train_counter.append(
                    (batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset)))
                torch.save(network.state_dict(), '../saved_networks/MNIST/model.pth')
                torch.save(optimizer.state_dict(), '../saved_networks/MNIST/optimizer.pth')




def test(network,test_loader,criterion):
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
              output = network(data)
              # test_loss += F.nll_loss(output, target, size_average=False).item()
              test_loss += criterion(output, target).item()
              pred = output.data.max(1, keepdim=True)[1]
              print("output.shape: ",output.shape)
              print("pred: ",pred)
              print("pred.shape:", pred.shape)
              correct += pred.eq(target.data.view_as(pred)).sum()
        test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))



network=Net()

# optimizer=torch.optim.Adam(network.parameters())
optimizer = torch.optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
# criterion=F.nll_loss()
criterion=torch.nn.NLLLoss()

# example_data, example_targets



output = network.forward(example_data[0:1])
loss = criterion(output, example_targets[0:1])

print("example_targets[0:1]: ",example_targets[0:1])
print("example_data[0:1].shape: ",example_data[0:1].shape)
print("output: ",output)
print("output: ",output.exp())
print("loss:", loss)
print("loss.item(): ",loss.item())





# train(n_epochs,optimizer,network,criterion)
# test(network,test_loader,criterion)





