import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import shutil
import os
import wandb


def setup_tensorboard(log_dir):
    """
    Sets up TensorBoard, ensuring the log directory exists and is clean.

    Args:
        log_dir (str): The directory where TensorBoard logs will be stored.
    Returns:
        SummaryWriter: A TensorBoard SummaryWriter instance.
    """

    # Clean the log directory if it exists
    if os.path.exists(log_dir):
        try:
            shutil.rmtree(log_dir)  # Remove the directory and its contents
            print(f"Removed existing TensorBoard logs at: {log_dir}")
        except OSError as e:
            print(f"Error removing existing TensorBoard logs: {e}")

    # Create the log directory
    os.makedirs(log_dir, exist_ok=True)
    print(f"TensorBoard logs will be saved to: {log_dir}")

    # Initialize the SummaryWriter
    writer = SummaryWriter(log_dir=log_dir)
    return writer


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # flatten the image
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# Get the current script path
current_script_path = Path(__file__).resolve()

# Go two directories above
two_directories_above = current_script_path.parent.parent

data_dir = str(two_directories_above)+'/data'

print("Current script:", current_script_path)
print("Two directories above:", two_directories_above)
print("data dir:", data_dir)


transform = transforms.ToTensor()
dataset = datasets.MNIST(root=data_dir, train=True,
                         download=True, transform=transform)

train_size = int(0.8*len(dataset))
validation_size = len(dataset)-train_size
train_set, val_set = random_split(dataset, [train_size, validation_size])


train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader = DataLoader(val_set, batch_size=64)


print(dataset)


def train(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        # print("loss: ", loss)
        # print("loss.item(): ", loss.item())
        # print("outputs.shape: ", outputs.shape)
        # print("outputs: ", outputs)
        # print("outputs.argmax(1): ", outputs.argmax(1))

        # print("labels: ", labels)
        # print("outputs.argmax(1) == labels: ", outputs.argmax(1) == labels)

        # print("(outputs.argmax(1) == labels).sum().item(): ",
        #       (outputs.argmax(1) == labels).sum().item())

        # exit()

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()

    accuracy = correct / len(loader.dataset)
    return total_loss / len(loader), accuracy


model = SimpleNN().to(device)


# for name, param in model.named_parameters():
#     print(f"{name}: {param.shape}")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


num_epochs = 5

train_losses = []
val_losses = []
val_accuracies = []


best_value_loss = float('inf')
patience = 3
epochs_without_improvement = 0

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')


# tensorboard --logdir=tensorboard_logs
log_dir = str(two_directories_above)+'/tensorboard_logs/'
writer = setup_tensorboard(log_dir)

wandb.init(mode="offline", project="local-test")

for epoch in range(num_epochs):
    train_loss = train(model, train_loader, criterion, optimizer, device)
    train_losses.append(train_loss)

    val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

    scheduler.step(val_loss)

    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | "
          f"Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy:.4f}")

    if val_loss < best_value_loss:
        best_value_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        epochs_without_improvement = epochs_without_improvement+1

    if epochs_without_improvement >= patience:
        print("Early stopping!")
        break

    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/val', val_loss, epoch)
    writer.add_scalar('Accuracy/val', val_accuracy, epoch)

    wandb.log({"accuracy": epoch * 0.1, "loss": 1 - epoch * 0.1})

writer.close()

epochs = [epoch for epoch in range(num_epochs)]

plt.plot(epochs, train_losses, color='r', label="train losses")
plt.plot(epochs, val_losses, color='b', label="val losses")
plt.plot(epochs, val_accuracies, color='g', label="val accuracies")
plt.legend(loc="best")
plt.grid(True)
plt.show()
