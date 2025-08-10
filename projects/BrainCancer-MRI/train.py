import argparse
import yaml
import os
import time
from utils.path_utils import get_project_root

from models.model import get_model
from data.dataset import load_datasets
from utils.helpers import save_checkpoint
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torchvision.models as models


def main(config_path):
    # Resolve absolute path to config file
    if not os.path.isabs(config_path):
        # Resolve relative to the script's directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, config_path)

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    weights = eval(config['model']['weights']
                   ) if config['model']['weights'] else None

    model, classifier_params = get_model(
        config['model']['name'],
        config['model']['num_classes'],
        weights
    )

    train_ds, val_ds, _ = load_datasets(config)

    train_loader = DataLoader(train_ds, batch_size=config['dataset']['batch_size'],
                              shuffle=True, num_workers=config['dataset']['num_workers'],
                              pin_memory=config['dataset']['pin_memory'])
    val_loader = DataLoader(val_ds, batch_size=config['dataset']['batch_size'],
                            shuffle=False, num_workers=config['dataset']['num_workers'])

    optimizer = optim.Adam(classifier_params, lr=config['model']['lr'])
    criterion = nn.CrossEntropyLoss()

    # Print training configuration
    print("="*60)
    print("🚀 Starting Brain Cancer MRI Training")
    print("="*60)
    print(
        f"📊 Dataset: {len(train_ds)} train, {len(val_ds)} validation samples")
    print(
        f"🏗️  Model: {config['model']['name']} with {config['model']['num_classes']} classes")
    print(f"📦 Batch size: {config['dataset']['batch_size']}")
    print(f"📈 Learning rate: {config['model']['lr']}")
    print(f"🔄 Epochs: {config['train']['epochs']}")
    print("="*60)

    for epoch in range(config['train']['epochs']):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        epoch_start_time = time.time()

        print(f"\n📚 Epoch {epoch+1}/{config['train']['epochs']}")
        print("-" * 50)

        for batch_idx, (x, y) in enumerate(train_loader):
            preds = model(x)
            loss = criterion(preds, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate metrics
            train_loss += loss.item()
            _, predicted = torch.max(preds.data, 1)
            train_total += y.size(0)
            train_correct += (predicted == y).sum().item()

            # Print progress every 10 batches
            if (batch_idx + 1) % 10 == 0:
                current_loss = train_loss / (batch_idx + 1)
                current_acc = 100. * train_correct / train_total
                print(f"  Batch {batch_idx+1}/{len(train_loader)} | "
                      f"Loss: {current_loss:.4f} | "
                      f"Acc: {current_acc:.2f}%")

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for x, y in val_loader:
                preds = model(x)
                loss = criterion(preds, y)
                val_loss += loss.item()
                _, predicted = torch.max(preds.data, 1)
                val_total += y.size(0)
                val_correct += (predicted == y).sum().item()

        # Calculate epoch metrics
        epoch_time = time.time() - epoch_start_time
        train_loss_avg = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        val_loss_avg = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total

        # Print epoch summary
        print(f"\n📊 Epoch {epoch+1} Summary:")
        print(f"  🕐 Time: {epoch_time:.2f}s")
        print(
            f"  📉 Train Loss: {train_loss_avg:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  📊 Val Loss: {val_loss_avg:.4f} | Val Acc: {val_acc:.2f}%")

        # Save checkpoint
        if epoch % config['train']['save_every'] == 0:
            print(f"💾 Saving checkpoint at epoch {epoch+1}")
            save_checkpoint(model, epoch, config)

    print("\n" + "="*60)
    print("🎉 Training completed successfully!")
    print("="*60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to config file (default: config/config.yaml)')
    args = parser.parse_args()
    main(args.config)
