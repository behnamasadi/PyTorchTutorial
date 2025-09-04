import argparse
from utils.file_utils import resource_path
import torch
import yaml
import tqdm
import models.model
import wandb
import mlflow


device = "cuda" if torch.cuda.is_available() else "cpu"
seed = 42


parser = argparse.ArgumentParser(
    description="Train brain MRI classification model",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument(
    "-c", "--config",
    default=resource_path('config', 'config.yaml'),
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
parser.add_argument(
    "--seed",
    type=int,
    default=42,
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

    # Get model and batch size from args or config
    model = args.model_name if args.model_name else config.get(
        "model", "default_model")
    batch_size = args.batch_size if args.batch_size else config.get(
        "batch_size", 32)

    # Get other training parameters
    epochs = args.epochs if args.epochs else config.get("epochs", 10)
    learning_rate = args.learning_rate if args.learning_rate else config.get(
        "learning_rate", 0.001)

    print(f"Model: {model}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {learning_rate}")
