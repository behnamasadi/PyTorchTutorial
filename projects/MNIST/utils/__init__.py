"""
Utility functions for the MNIST project
"""

import os
import json
import torch
import yaml
from datetime import datetime


def save_experiment(model, hyperparameters, metrics, save_dir='experiments'):
    """
    Save model, hyperparameters, and metrics for reproducibility.

    Args:
        model: PyTorch model
        hyperparameters: Dictionary of hyperparameters
        metrics: Dictionary of metrics
        save_dir: Directory to save the experiment
    """
    # Create timestamp for unique experiment ID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(save_dir, f"experiment_{timestamp}")

    # Create directory if it doesn't exist
    os.makedirs(experiment_dir, exist_ok=True)

    # Save model
    model_path = os.path.join(experiment_dir, "model.pt")
    torch.save(model.state_dict(), model_path)

    # Save hyperparameters
    hyperparams_path = os.path.join(experiment_dir, "hyperparameters.json")
    with open(hyperparams_path, 'w') as f:
        json.dump(hyperparameters, f, indent=4)

    # Save metrics
    metrics_path = os.path.join(experiment_dir, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)

    print(f"Experiment saved to {experiment_dir}")
    return experiment_dir


def load_experiment(experiment_dir, model_class):
    """
    Load a saved experiment.

    Args:
        experiment_dir: Directory containing the experiment
        model_class: The model class to instantiate

    Returns:
        model: Loaded PyTorch model
        hyperparameters: Dictionary of hyperparameters
        metrics: Dictionary of metrics
    """
    # Load hyperparameters
    hyperparams_path = os.path.join(experiment_dir, "hyperparameters.json")
    with open(hyperparams_path, 'r') as f:
        hyperparameters = json.load(f)

    # Load metrics
    metrics_path = os.path.join(experiment_dir, "metrics.json")
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    # Create model with saved hyperparameters
    model = model_class(**hyperparameters['model_params'])

    # Load model weights
    model_path = os.path.join(experiment_dir, "model.pt")
    model.load_state_dict(torch.load(model_path))

    return model, hyperparameters, metrics


def save_config(config, save_path):
    """
    Save configuration to YAML file.

    Args:
        config: Dictionary of configuration parameters
        save_path: Path to save the configuration
    """
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"Configuration saved to {save_path}")


def load_config(config_path):
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to the configuration file

    Returns:
        config: Dictionary of configuration parameters
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config
