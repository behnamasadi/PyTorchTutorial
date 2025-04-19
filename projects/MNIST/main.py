import os
import torch
import random
import numpy as np
import wandb
from models.mlp import MLP
from models.resnet_mlp import ResNetMLP
from data.mlp_dataloader import MLPMNISTDataLoader
from data.resnet_dataloader import ResNetMNISTDataLoader
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


def get_model(model_type, config):
    """Initialize the appropriate model based on model_type"""
    if model_type == 'mlp':
        return MLP(
            input_dim=config['model']['mlp']['input_dim'],
            hidden_dim=config['model']['mlp']['hidden_dim'],
            output_dim=config['model']['mlp']['output_dim'],
            dropout=config['model']['mlp']['dropout']
        )
    elif model_type == 'resnet':
        return ResNetMLP()
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def get_dataloader(model_type, config):
    """Get the appropriate dataloader based on model_type"""
    if model_type == 'mlp':
        loader = MLPMNISTDataLoader(
            batch_size=config['data']['batch_size'],
            num_workers=config['data']['num_workers']
        )
    elif model_type == 'resnet':
        loader = ResNetMNISTDataLoader(
            batch_size=config['data']['batch_size'],
            num_workers=config['data']['num_workers']
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return loader


def main():
    # Load configuration
    config_path = "configs/default.yaml"
    config = load_config(config_path)

    # Get model type from config
    model_type = config['model']['type']
    print(f"Using model type: {model_type}")

    # Set random seed for reproducibility
    set_seed(config['experiment']['seed'])

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Get appropriate dataloader
    dataloader = get_dataloader(model_type, config)

    # Get data loaders with train/val split
    train_loader, val_loader = dataloader.get_mnist_dataloaders(
        train=True,
        val_split=config['data']['val_split']
    )

    test_loader, = dataloader.get_mnist_dataloaders(
        train=False
    )

    # Initialize model
    model = get_model(model_type, config)
    model = model.to(device)

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
        'model_type': model_type,
        'model_params': config['model'][model_type],
        'training_params': {
            'learning_rate': config['training']['learning_rate'],
            'weight_decay': config['training']['weight_decay'],
            'num_epochs': config['training']['num_epochs'],
            'patience': config['training']['patience'],
            'scheduler': config['training']['scheduler']
        },
        'data_params': {
            'batch_size': config['data']['batch_size'],
            'val_split': config['data']['val_split'],
            'num_workers': config['data']['num_workers']
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
        f"Training completed. Best validation accuracy: "
        f"{trainer.best_val_acc:.2f}%"
    )
    print(f"Final test accuracy: {metrics['final_test_acc']:.2f}%")
    print(f"Experiment saved to: {experiment_dir}")

    # Finish wandb run
    wandb.finish()


if __name__ == "__main__":
    main()
