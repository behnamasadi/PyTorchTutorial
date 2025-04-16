import os
import argparse
import torch
from models.mlp import MLP
from data import get_mnist_dataloaders
from utils import load_experiment, load_config


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Evaluate a saved MNIST model')
    parser.add_argument(
        '--experiment_dir',
        type=str,
        required=True,
        help='Directory containing the saved experiment'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Batch size for evaluation'
    )
    args = parser.parse_args()

    # Load experiment
    model, hyperparameters, metrics = load_experiment(args.experiment_dir, MLP)

    # Load configuration (for reference)
    config_path = os.path.join(args.experiment_dir, "config.yaml")
    _ = load_config(config_path)  # Load but not used directly

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Move model to device
    model = model.to(device)

    # Get test dataloader
    test_loader, = get_mnist_dataloaders(
        batch_size=args.batch_size,
        train=False
    )

    # Evaluate model
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    accuracy = 100. * correct / total

    # Print results
    print(f"Model from experiment: {args.experiment_dir}")
    print(f"Original test accuracy: {metrics['final_test_acc']:.2f}%")
    print(f"Current test accuracy: {accuracy:.2f}%")

    # Print hyperparameters
    print("\nModel hyperparameters:")
    for key, value in hyperparameters['model_params'].items():
        print(f"  {key}: {value}")

    print("\nTraining hyperparameters:")
    for key, value in hyperparameters['training_params'].items():
        if key != 'scheduler':
            print(f"  {key}: {value}")

    print("\nData hyperparameters:")
    for key, value in hyperparameters['data_params'].items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
