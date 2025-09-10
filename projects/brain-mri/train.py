import argparse
import torch
import yaml
import pathlib
import tqdm


from torch.utils.data import DataLoader, Dataset, random_split
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
import torchvision.models as models


from models.model import get_model
from utils.file_utils import resource_path

import matplotlib.pyplot as plt

# import wandb
# import mlflow


device = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed=seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed=seed)
        torch.cuda.manual_seed_all(seed=seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return


parser = argparse.ArgumentParser(
    description="Train brain MRI classification model",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument(
    "-c", "--config",
    default=resource_path('config', 'config_dev.yaml'),
    help="Path to configuration file"
)
parser.add_argument(
    "-m", "--model_name",
    type=str,
    help="Name of the model to use"
)
parser.add_argument(
    "-b", "--batch_size",
    type=int,
    help="Batch size for training"
)
parser.add_argument(
    "-e", "--epochs",
    type=int,
    help="Number of training epochs"
)
parser.add_argument(
    "-lr", "--learning_rate",
    type=float,
    help="Learning rate for optimizer"
)
parser.add_argument(
    "--device",
    type=str,
    default=device,
    help="Device to use for training (cuda/cpu)"
)


args = parser.parse_args()

if __name__ == "__main__":

    # Set device and seed from arguments
    device = args.device

    print(f"Using device: {device}")

    # Load configuration
    print(f"Using config: {args.config}")

    cfg = args.config
    print(f"Using config: {cfg}")
    with open(cfg) as file_cfg:
        config = yaml.safe_load(file_cfg)

    selected_model_name = config.get("selected_model")
    config_all_models = config.get("models")

    # Get model and batch size from args or config
    selected_model = args.model_name if args.model_name else config_all_models.get(
        selected_model_name)

    print(f"Selected model name: {selected_model}")

    batch_size = args.batch_size if args.batch_size else selected_model.get(
        "batch_size")

    # just use weight = true and load them in class
    weights_str = selected_model.get("weights")
    # Convert string to actual weights enum
    if weights_str == "models.EfficientNet_B0_Weights.IMAGENET1K_V1":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
    elif weights_str == "torch.MobileNet_V2_Weights":
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
    else:
        raise ValueError(f"Unknown weights: {weights_str}")

    num_classes = selected_model.get("num_classes")

    # Get other training parameters
    epochs = args.epochs if args.epochs else selected_model.get("epochs")
    learning_rate = args.learning_rate if args.learning_rate else selected_model.get(
        "lr")

    data_stats = config.get("data_stats")
    seed = data_stats.get("seed")
    mean = list(data_stats.get("mean"))
    std = list(data_stats.get("std"))

    data_path = data_stats.get("data_path")
    # Ensure proper path handling for cross-platform compatibility
    full_path_data = resource_path(data_path)
    # Convert to string for torchvision.datasets.ImageFolder which expects string paths
    full_path_data_str = str(full_path_data.resolve())
    print(f"Resolved data path: {full_path_data_str}")

    train_split = data_stats.get("train_split")
    val_split = data_stats.get("val_split")
    img_size = data_stats.get("img_size")

    print(f"Full Path Data: {full_path_data_str}")
    print(f"Train Split: {train_split}")
    print(f"Val Split: {val_split}")
    print(f"Random seed: {seed}")
    print(f"mean: {mean}")
    print(f"std: {std}")
    print(f"Selected model name: {selected_model_name}")
    print(f"Weights: {weights}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Image size: {img_size}")

    full_dataset = torchvision.datasets.ImageFolder(root=full_path_data_str, transform=transforms.Compose([
        transforms.RandAugment(),
        transforms.Resize(256),  # resize shortest side to 256

        transforms.CenterCrop(img_size),  # take center 224x224
        transforms.ColorJitter(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)]
    ))

    dataset_length = len(full_dataset)
    train_length = int(dataset_length*float(train_split))
    val_length = int(dataset_length*float(val_split))

    test_length = dataset_length-(train_length+val_length)

    print(f"Train Length: {train_length}")
    print(f"Val Length: {val_length}")
    print(f"Test Length: {test_length}")

    train_set, val_set, test_set = random_split(dataset=full_dataset, lengths=[
                                                train_length, val_length, test_length], generator=torch.Generator())

    train_loader = DataLoader(batch_size=batch_size,
                              dataset=train_set, shuffle=True, pin_memory=True)
    val_loader = DataLoader(batch_size=batch_size,
                            dataset=val_set)
    test_loader = DataLoader(batch_size=batch_size,
                             dataset=test_set)

    model = get_model(model_name=selected_model_name,
                      weights=weights, num_classes=num_classes)

    print(model)

    # print(model.named_parameters["features"])
    # features.8.1.weight
    # features.8.1.bias
    # classifier.1.weight
    # classifier.1.bias

    # print(model.classifier[1])

    # For MRI images, we need more flexibility than just classifier training
    # Option 1: Unfreeze only the last few layers of the backbone
    for param in model.parameters():
        param.requires_grad = False

    # # Unfreeze the last few layers of the backbone (more relevant for medical images)
    # if hasattr(model, 'backbone') and hasattr(model.backbone, 'features'):
    #     # For EfficientNet, unfreeze the last 2-3 blocks
    #     for name, param in model.backbone.features.named_parameters():
    #         if any(block in name for block in ['7', '8']):  # Last 2 blocks
    #             param.requires_grad = True
    #             print(f"Unfrozen: {name}")

    # Always unfreeze classifier
    for name, param in model.named_parameters():
        if 'classifier' in name:
            param.requires_grad = True
            print(f"Unfrozen classifier: {name}")

    print("\nTrainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  {name}: {param.shape}")

    # exit()
    model.to(device)
    model.train()

    patience = 10
    criterion = torch.nn.CrossEntropyLoss(
        label_smoothing=0.1, reduction="mean")
    optimizer = torch.optim.Adamax(lr=learning_rate, params=model.parameters())

    loss_train = []
    acc_train = []
    loss_val = []
    acc_val = []

    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(epochs):
        print(f"Epoch number: ", epoch)

        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for images, targets in train_loader:
            # batch size x class number 128 x 3
            # targets is actual_class_index

            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            predict = model(images)
            loss = criterion(predict, targets)
            loss.backward()
            optimizer.step()

            # preds is predicted_classes_index
            preds = torch.argmax(predict, 1)
            running_loss += loss.item() * images.size(0)

            correctly_predicted = (targets == preds)
            running_correct += correctly_predicted.sum().item()
            running_total += images.size(0)

        epoch_loss = running_loss/running_total
        epoch_acc = running_correct/running_total
        print(
            f"Epoch {epoch}: Train loss={epoch_loss:.4f}, Train acc={epoch_acc:.4f}")

        loss_train.append(epoch_loss)
        acc_train.append(epoch_acc)

        # Validation loop
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                outputs = model(images)
                loss = criterion(outputs, targets)

                val_loss += loss.item() * images.size(0)
                preds = torch.argmax(outputs, 1)
                val_correct += (preds == targets).sum().item()
                val_total += images.size(0)

        val_epoch_loss = val_loss / val_total
        val_epoch_acc = val_correct / val_total
        print(
            f"Epoch {epoch}: Val loss={val_epoch_loss:.4f}, Val acc={val_epoch_acc:.4f}")

        loss_val.append(val_epoch_loss)
        acc_val.append(val_epoch_acc)

        # Early stopping
        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

        # Plot every 5 epochs to avoid too many plots
        if epoch % 5 == 0 or epoch == epochs - 1:
            plt.figure(figsize=(15, 5))

            plt.subplot(1, 3, 1)
            plt.plot(range(len(loss_train)), loss_train,
                     'b-', label='Training Loss')
            plt.plot(range(len(loss_val)), loss_val,
                     'r-', label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Loss')
            plt.legend()

            plt.subplot(1, 3, 2)
            plt.plot(range(len(acc_train)), acc_train,
                     'b-', label='Training Accuracy')
            plt.plot(range(len(acc_val)), acc_val,
                     'r-', label='Validation Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.title('Accuracy')
            plt.legend()

            plt.subplot(1, 3, 3)
            plt.plot(range(len(acc_train)), acc_train,
                     'b-', label='Training Accuracy')
            plt.plot(range(len(acc_val)), acc_val,
                     'r-', label='Validation Accuracy')
            plt.axhline(y=best_val_acc, color='g', linestyle='--',
                        label=f'Best Val Acc: {best_val_acc:.4f}')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.title('Accuracy with Best')
            plt.legend()

            plt.tight_layout()
            plt.show()
