import argparse
import yaml
import os
from utils.path_utils import get_project_root
from models.model import get_model
from data.dataset import load_datasets
from utils.helpers import save_checkpoint
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn


def main(config_path):
    if not os.path.isabs(config_path):
        config_path = os.path.join(get_project_root(), config_path)

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

    for epoch in range(config['train']['epochs']):
        model.train()
        for x, y in train_loader:
            preds = model(x)
            loss = criterion(preds, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} done")

        if epoch % config['train']['save_every'] == 0:
            save_checkpoint(model, epoch, config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/config.yaml')
    args = parser.parse_args()
    main(args.config)
