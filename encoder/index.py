# https://www.youtube.com/watch?v=9zKuYvjFFS8&ab_channel=ArxivInsights
# https://medium.com/pytorch/implementing-an-autoencoder-in-pytorch-19baa22647d1
# https://gist.github.com/AFAgarap/4f8a8d8edf352271fa06d85ba0361f26
import torch
import torchvision
import matplotlib.pyplot as plt
import torchviz


class AE(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        in_features = kwargs["input_shape"]

        self.encoder_hidden_layer = torch.nn.Linear(in_features=in_features, out_features=128)
        self.encoder_output_layer = torch.nn.Linear(in_features=128, out_features=128)

        self.decoder_hidden_layer = torch.nn.Linear(in_features=128, out_features=128)
        self.decoder_output_layer = torch.nn.Linear(in_features=128, out_features=in_features)

    def forward(self , features):
        activations = self.encoder_hidden_layer(features)
        activations = self.encoder_output_layer(activations)
        code = torch.relu(activations)

        activation = self.decoder_hidden_layer(code)
        activation = torch.relu(activation)
        activation = self.decoder_output_layer(activation)
        reconstructed = torch.relu(activation)
        return reconstructed


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)


# 28 by 28 pixels of MNIST=784
model = AE(input_shape=784).to(device)


input = torch.randn(size=[1,784]).to(device)
dot = torchviz.make_dot(model(input),params=dict(model.named_parameters()) )
dot.format='svg'
dot.render(filename='simple_encoder_decoder_graph', directory='../images')

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.MSELoss()


data_transoformer=torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

# [60000, 28, 28].
train_data=torchvision.datasets.MNIST(root='../data',train=True, transform=data_transoformer, download=True)

test_data=torchvision.datasets.MNIST(root='../data',train=False, transform=data_transoformer, download=True)


train_loader = torch.utils.data.DataLoader( train_data, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)

test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False, num_workers=4)


epochs=1

for epoch in range(epochs):
    loss = 0
    for batch_features, _ in train_loader:
        # reshape mini-batch data to [N, 784] matrix
        # load it to the active device
        batch_features = batch_features.view(-1, 784).to(device)

        # reset the gradients back to zero
        # PyTorch accumulates gradients on subsequent backward passes
        optimizer.zero_grad()

        # compute reconstructions
        outputs = model(batch_features)

        # compute training reconstruction loss
        train_loss = criterion(outputs, batch_features)

        # compute accumulated gradients
        train_loss.backward()

        # perform parameter update based on current gradients
        optimizer.step()

        # add the mini-batch training loss to epoch loss
        loss += train_loss.item()

    # compute the epoch training loss
    # len(train_loader) -> 60000/128=429
    loss = loss / len(train_loader)

    # display the epoch training loss
    print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, loss))





test_examples = None
# we set the batch_size 32 so len(batch_features) is 32
with torch.no_grad():
    for batch_features in test_loader:
        batch_features = batch_features[0]
        test_examples = batch_features.view(-1, 784).to(device)
        reconstruction = model(test_examples)
        # we only do it for one iteration for 32 images
        break

with torch.no_grad():
    number = 10
    plt.figure(figsize=(20, 4))
    for index in range(number):
        # display original
        ax = plt.subplot(2, number, index + 1)
        plt.imshow(test_examples[index].cpu().numpy().reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, number, index + 1 + number)
        plt.imshow(reconstruction[index].cpu().numpy().reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()




