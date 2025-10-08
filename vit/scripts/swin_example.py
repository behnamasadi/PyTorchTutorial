import math
import time
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import datasets, transforms
import torch.optim as optim
import timm

import matplotlib.pyplot as plt
from utils.file_utils import resource_path


device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

data_dir = "data"
img_size = 224
batch_size = 192  # Try: 160, 192, 224 (use multiples of 8 or powers of 2)
workers = 2
epochs = 20
num_classes = 10

interval_update_plot = 5

# If you hit OOM (Out of Memory), try gradient accumulation instead:
# accumulation_steps = 2  # Effective batch size = batch_size * accumulation_steps

argument_parser = argparse.ArgumentParser(
    description="provide arguments for swin transformer ")
argument_parser.add_argument(
    "--epochs", "-e", default="1", help="number of epochs")

args = argument_parser.parse_args()

# if (args.epoch):
#     print(args.epoch)

train_set_percentage, val_set_percentage = 0.85, 0.15


# CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class

CIFAR10_root_data = "data"
print(resource_path(CIFAR10_root_data))


tfm_train = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                         std=(0.2023, 0.1994, 0.2010)),
])


tfm_test = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                         std=(0.2023, 0.1994, 0.2010)),
])

train_CIFAR10_full_dataset = datasets.CIFAR10(
    download=True, root=CIFAR10_root_data, train=True, transform=tfm_train)

test_CIFAR10_dataset = datasets.CIFAR10(
    download=True, root=CIFAR10_root_data, train=False, transform=tfm_test)

train_CIFAR10_full_dataset_len = len(train_CIFAR10_full_dataset)
print(train_CIFAR10_full_dataset_len)


test_CIFAR10_dataset_len = len(test_CIFAR10_dataset)
print(test_CIFAR10_dataset_len)


train_set_len = int(train_set_percentage*train_CIFAR10_full_dataset_len)
val_set_len = train_CIFAR10_full_dataset_len-train_set_len


# print([train_set_len, val_set_len])
# exit()

train_set, val_set = random_split(
    lengths=[train_set_len, val_set_len], dataset=train_CIFAR10_full_dataset)

train_set_dataloader = DataLoader(
    batch_size=batch_size, pin_memory=True, dataset=train_set)


val_set_dataloader = DataLoader(
    batch_size=batch_size, pin_memory=True, dataset=val_set)


test_set_dataloader = DataLoader(
    batch_size=batch_size, pin_memory=True, dataset=test_CIFAR10_dataset)


model = timm.create_model("swin_tiny_patch4_window7_224",
                          pretrained=True, num_classes=num_classes)


model.to(device)
for p in model.parameters():
    p.requires_grad = False

for p in model.get_classifier().parameters():
    p.requires_grad = True

optimizer = optim.AdamW(params=filter(
    lambda p: p.requires_grad, model.parameters()), lr=0.005)

criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1, reduction="mean")

# Mixed Precision Training Setup
use_amp = True if device == "cuda" else False
device_type = 'cuda' if device == 'cuda' else 'cpu'
scaler = torch.amp.GradScaler(device_type, enabled=use_amp)
print(
    f"Mixed Precision Training (AMP): {'Enabled' if use_amp else 'Disabled'}")

loss_train = []
loss_val = []

acc_train = []
acc_val = []

for epoch in range(epochs):
    print(epoch)
    model.train()
    running_loss = 0
    running_correct = 0
    running_total = 0
    for images, labels in train_set_dataloader:

        images, labels = images.to(device, non_blocking=True), labels.to(
            device, non_blocking=True)
        batch_size = images.size(0)

        optimizer.zero_grad(set_to_none=True)

        # Mixed precision training
        with torch.amp.autocast(device_type=device_type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        preds = torch.argmax(logits, 1)
        running_loss += loss.item() * batch_size

        correctly_predicted = (preds == labels)
        running_correct = running_correct+correctly_predicted.sum().item()
        running_total = running_total+batch_size

    epoch_loss = running_loss/running_total
    epoch_acc = running_correct/running_total

    print(
        f"Epoch {epoch}: Train loss={epoch_loss:.4f}, Train acc={epoch_acc:.4f}")

    acc_train.append(epoch_acc)
    loss_train.append(epoch_loss)

    # Validation
    running_correct_val = 0
    running_loss_val = 0
    running_total_val = 0

    model.eval()
    with torch.no_grad():

        for images, labels in val_set_dataloader:

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            batch_size = images.size(0)

            # Use autocast for validation too
            with torch.amp.autocast(device_type=device_type, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)

            preds = torch.argmax(logits, dim=1)
            correctly_predict = (preds == labels)
            running_correct_val = running_correct_val + correctly_predict.sum().item()

            running_loss_val = running_loss_val + loss.item() * batch_size
            running_total_val = running_total_val + batch_size

    epoch_loss_val = running_loss_val / running_total_val
    epoch_acc_val = running_correct_val / running_total_val

    print(
        f"Epoch {epoch}: Val loss={epoch_loss_val:.4f}, Val acc={epoch_acc_val:.4f}")

    acc_val.append(epoch_acc_val)
    loss_val.append(epoch_loss_val)

    # Plot every interval_update_plot epochs or at the last epoch
    if ((epoch + 1) % interval_update_plot == 0) or (epoch == epochs - 1):
        plt.figure(figsize=(12, 5))

        # Plot Loss
        plt.subplot(1, 2, 1)
        current_epochs = range(1, epoch + 2)
        plt.plot(current_epochs, loss_train, label='Train Loss',
                 marker='o', linewidth=2, markersize=4)
        plt.plot(current_epochs, loss_val, label='Val Loss',
                 marker='s', linewidth=2, markersize=4)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Training and Validation Loss',
                  fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)

        # Plot Accuracy
        plt.subplot(1, 2, 2)
        plt.plot(current_epochs, acc_train, label='Train Accuracy',
                 marker='o', linewidth=2, markersize=4)
        plt.plot(current_epochs, acc_val, label='Val Accuracy',
                 marker='s', linewidth=2, markersize=4)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title('Training and Validation Accuracy',
                  fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            f'training_curves_epoch_{epoch+1}.png', dpi=300, bbox_inches='tight')
        print(f"Plot saved to 'training_curves_epoch_{epoch+1}.png'")
        plt.show()
        plt.close()

# Final summary plot
print("\nTraining complete!")
print(f"Final Train Accuracy: {acc_train[-1]:.4f}")
print(f"Final Val Accuracy: {acc_val[-1]:.4f}")
