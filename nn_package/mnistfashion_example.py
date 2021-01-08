import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append("../utility")
import shape_calculator

def images_to_probability(net, images):
    output=net(images)

    # the output is a tensor of size: [number of batch x number of classes] ->    4x10
    # we want to find the index of max value in each row so dim=1
    _,preds_tensor_index=torch.max(output,dim=1)

    preds=np.squeeze(preds_tensor_index.numpy())

    # here we iterating over each rows in output and computing softmax for each row (dim=0 when we iterating
    # over rows of output and not over complete tensor )
    # the return values are index of max value and probability of corresponding class
    # so the total number of items in return is: number of batch
    return preds, [torch.nn.functional.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]


def plot_classes_preds(net, images, labels):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    preds, probs = images_to_probs(net, images)
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(12, 48))
    for idx in np.arange(4):
        ax = fig.add_subplot(1, 4, idx+1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx], one_channel=True)
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            classes[preds[idx]],
            probs[idx] * 100.0,
            classes[labels[idx]]),
                    color=("green" if preds[idx]==labels[idx].item() else "red"))
    return fig


def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

def select_n_random(data, labels, n=100):
    '''select n random images and their corresponding labels'''
    # data is mnist_data_train.data which is of type torch.Tensor and shape is torch.Size([60000, 28, 28])

    # perm would be a shuffle of numbers between 0 to 59999
    perm=torch.randperm(len(data))

    # [:n] will return the first n elements of an array, we want the first n element of data[perm] so data[perm][:n]
    return data[perm][:n], labels[perm][:n]

# images are one channel and 28x28





# transforms
pre_proc=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                         torchvision.transforms.Normalize((0.5,), (0.5,))])

# datasets
mnist_data_train=torchvision.datasets.FashionMNIST(root="../data/MNIST",
                                                   download=True,
                                                   transform=pre_proc,
                                                   train=True)

mnist_data_test=torchvision.datasets.FashionMNIST(root="../data/MNIST",
                                                  download=True,
                                                  transform=pre_proc,
                                                  train=False)
# dataloaders

batch_size=4

mnist_data_train_loader=torch.utils.data.DataLoader(mnist_data_train,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    num_workers=4)

mnist_data_train_iter=iter(mnist_data_train_loader)
images_train,labels_train=mnist_data_train_iter.next()

mnist_data_test_loader=torch.utils.data.DataLoader(mnist_data_test,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=4)

class MyCustomNet(torch.nn.Module):
    def __init__(self):
        super(MyCustomNet,self).__init__()
        self.cov1 = torch.nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5)
        self.cov2 = torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)

        height=28
        width=28
        print("original image height, width", height, width)
        kernel_size = (5, 5)
        stride = (1, 1)
        padding = (0, 0)
        dilation = (1, 1)

        # conv1
        height,width=shape_calculator.conv2d(height, width, padding, dilation, kernel_size, stride)
        print("after conv1 height, width", height, width)

        # first max pool
        kernel_size = (2,2)
        stride = (2,2)
        height,width=shape_calculator.maxPool2d(height, width, padding, dilation, kernel_size, stride)
        print("afte first max pool height, width", height, width)


        # conv2
        kernel_size = (5, 5)
        stride = (1,1)
        height, width = shape_calculator.conv2d(height, width, padding, dilation, kernel_size, stride)
        print("after conv2 height, width", height, width)

        # second max pool
        kernel_size = (2, 2)
        stride = (2, 2)
        height, width = shape_calculator.maxPool2d(height, width, padding, dilation, kernel_size, stride)
        print("second max pool height, width", height, width)

        # at this stage, after two time conv and max pooling, the image height=4 width=4 and depth=16 (out_channels=16)

        self.fc1=torch.nn.Linear(16 * 4 * 4, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84,10)
        pass
    def forward(self,x):
        print("x.dtype", x.dtype)
        print("x.shape", x.shape)
        x=self.cov1(x)
        x = torch.nn.functional.max_pool2d(x,kernel_size=2, stride=2)
        x=torch.nn.functional.relu(x)

        x=self.cov2(x)
        x = torch.nn.functional.max_pool2d(x,kernel_size=2, stride=2)
        x=torch.nn.functional.relu(x)

        x = x.view(-1, 16 * 4 * 4)


        x=self.fc1(x)
        x=torch.nn.functional.relu(x)

        x=self.fc2(x)
        x = torch.nn.functional.relu(x)

        x=self.fc3(x)
        return x


# mnist_data_test_iter=iter(mnist_data_test_loader)
# images,labels=mnist_data_test_iter.next()

# constant for classes
#classes=['0','1','2','3','4','5','6','7','8','9']
classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')

# batch x channel x row x column
#images_train = images_train.view(images_train.size(0), 1, 28, 28)
model = MyCustomNet()
print("images_train.shape:",images_train.shape)
y = model(images_train)
print("y.shape:",y.shape)
print("y:\n",y)


print("softmax:\n",torch.nn.functional.softmax(y, dim=0))

loss_function=torch.nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(model.parameters(),momentum=0.9,lr=0.001)


#loss_function(y,labels_train)



from torch.utils.tensorboard import SummaryWriter
# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter('runs/fashion_mnist_experiment_1')
image_grid=torchvision.utils.make_grid(images_train)

matplotlib_imshow(image_grid,one_channel=True)
writer.add_image('four_fashion_mnist_images', image_grid)

print(type(images_train[0]))
print(images_train[0].shape)
writer.add_graph(model,images_train)
writer.close()


print("type(mnist_data_train):",type(mnist_data_train))
print(type(mnist_data_train[0]))
print(type(mnist_data_train[0][0]))
print(type(mnist_data_train[0][1]))
print("type(mnist_data_train.data):",type(mnist_data_train.data))
print("mnist_data_train.data.shape:",mnist_data_train.data.shape)
print(type(mnist_data_train.targets))
print(mnist_data_train.targets.shape)
print(mnist_data_train.targets)

# select random images and their target indices
random_images, random_labels = select_n_random(mnist_data_train.data, mnist_data_train.targets)

# get the class labels for each image
class_labels = [classes[lab] for lab in random_labels]
print(random_images.shape)
features=random_images.view(-1,28*28)
writer.add_embedding(features,metadata=class_labels,label_img=random_images.unsqueeze(1))
writer.close()




running_loss = 0.0
for epoch in range(1):  # loop over the dataset multiple times

    for i, data in enumerate(mnist_data_train, 0):

        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        print("type(inputs):", type(inputs))
        print("inputs.shape:", inputs.shape)
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 1000 == 999:    # every 1000 mini-batches...

            # ...log the running loss
            writer.add_scalar('training loss',
                            running_loss / 1000,
                            epoch * len(mnist_data_train) + i)

            # ...log a Matplotlib Figure showing the model's predictions on a
            # random mini-batch
            writer.add_figure('predictions vs. actuals',
                            plot_classes_preds(model, inputs, labels),
                            global_step=epoch * len(mnist_data_train) + i)
            running_loss = 0.0
print('Finished Training')