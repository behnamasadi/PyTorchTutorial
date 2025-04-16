import os
import torch
import random
import numpy as np
from models.mlp import MLP
from data import get_mnist_dataloaders
from trainers.base_trainer import BaseTrainer
from utils import save_experiment, load_config, save_config


def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    # Load configuration
    config_path = "configs/default.yaml"
    config = load_config(config_path)

    # Set random seed for reproducibility
    set_seed(config['experiment']['seed'])

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Get data loaders with train/val split
    train_loader, val_loader = get_mnist_dataloaders(
        batch_size=config['data']['batch_size'],
        train=True,
        val_split=config['data']['val_split']
    )

    test_loader, = get_mnist_dataloaders(
        batch_size=config['data']['batch_size'],
        train=False
    )

    # Initialize model
    model = MLP(
        input_dim=config['model']['input_dim'],
        hidden_dim=config['model']['hidden_dim'],
        output_dim=config['model']['output_dim'],
        dropout=config['model']['dropout']
    )

    # Initialize trainer
    trainer = BaseTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        num_epochs=config['training']['num_epochs'],
        patience=config['training']['patience'],
        device=device
    )

    # Start training
    trainer.train()

    # Collect metrics
    metrics = {
        'best_val_acc': trainer.best_val_acc,
        'best_epoch': trainer.best_epoch,
        'final_test_acc': trainer.validate(trainer.test_loader)[1]
    }

    # Collect hyperparameters
    hyperparameters = {
        'model_params': {
            'input_dim': config['model']['input_dim'],
            'hidden_dim': config['model']['hidden_dim'],
            'output_dim': config['model']['output_dim'],
            'dropout': config['model']['dropout']
        },
        'training_params': {
            'learning_rate': config['training']['learning_rate'],
            'weight_decay': config['training']['weight_decay'],
            'num_epochs': config['training']['num_epochs'],
            'patience': config['training']['patience'],
            'scheduler': config['training']['scheduler']
        },
        'data_params': {
            'batch_size': config['data']['batch_size'],
            'val_split': config['data']['val_split']
        }
    }

    # Save experiment
    save_dir = config['experiment']['save_dir']
    experiment_dir = save_experiment(
        model,
        hyperparameters,
        metrics,
        save_dir
    )

    # Save configuration to experiment directory
    config_save_path = os.path.join(experiment_dir, "config.yaml")
    save_config(config, config_save_path)

    print(
        f"Training completed. Best validation accuracy: {trainer.best_val_acc:.2f}%"
    )
    print(f"Final test accuracy: {metrics['final_test_acc']:.2f}%")
    print(f"Experiment saved to: {experiment_dir}")


if __name__ == "__main__":
    main()
