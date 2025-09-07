import argparse
from utils.file_utils import resource_path
import torch
import yaml
import tqdm
import models.model
from torchvision.datasets import ImageFolder


#from torch.utils.data import dataloader, dataset, random_split
from torch.utils.data import DataLoader, random_split,dataset


# import wandb
# import mlflow


device_default = "cuda" if torch.cuda.is_available() else "cpu"
seed_default =42
batch_size_default=64
epochs_default=20
learning_rate_default=0.001
patience_default=10
lengths=[70,15,15]

def validate_config(config):
    required_keys = ["models", "selected_model"]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")

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
    default=device_default,
    help="Device to use for training (cuda/cpu)"
)
parser.add_argument(
    "--seed",
    type=int,
    default=seed_default,
    help="Random seed for reproducibility"
)

args = parser.parse_args()

if __name__ == "__main__":
    # Set device and seed from arguments
    device = args.device
    seed = args.seed

    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    print(f"Using device: {device}")
    print(f"Random seed: {seed}")

    # Load configuration
    cfg = args.config
    print(f"Using config: {cfg}")
    with open(cfg) as file_cfg:
        config = yaml.safe_load(file_cfg)

    validate_config(config)

    # Get model and batch size from args or config
    # model_name = args.model_name if args.model_name else config.get("selected_model", "efficientnet_b0")
    model_name = args.model_name if args.model_name else config.get("selected_model")
    
    
    batch_size = args.batch_size if args.batch_size else config.get("models", {}).get(str(model_name), {}).get("batch_size", batch_size_default) 

    # Get other training parameters
    epochs = args.epochs if args.epochs else config.get("epochs", epochs_default)
    learning_rate = args.learning_rate if args.learning_rate else config.get(
        "learning_rate", learning_rate_default)

    print(f"Model: {model_name}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {learning_rate}")
    
    
    # Get number of classes from config or use default
    num_classes = config.get("models", {}).get(str(model_name), {}).get("num_classes", 3)
    
    # Initialize the model
    model = models.model.get_model(
        model_name=str(model_name),
        num_classes=num_classes,
        pretrained=True
    )
    
    # Move model to device
    model = model.to(device)
    
    print(f"Model loaded: {model.__class__.__name__}")
    print(f"Number of classes: {num_classes}")
    
    # p.numel() Returns the total number of elements in the input tensor
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    
    class brainImageFolder(ImageFolder) :
        def __init__(self, root, transform = None, target_transform = None, loader = ..., is_valid_file = None, allow_empty = False):
            super().__init__(root, transform, target_transform, loader, is_valid_file, allow_empty)
        
    
    
    # class brainImageDataset()
    datase
    random_split(dataset=)
    
    
    # dataloader.DataLoader( batch_size=batch_size, generator="",num_workers=,pin_memory=True, )
    
    for epoch in range(epochs):
        for batch_idx in range(batch_size):
            model.train()
            # print(".")
            pass

