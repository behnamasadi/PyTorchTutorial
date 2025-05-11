
import argparse
import yaml
import os
from utils.path_utils import get_project_root
from data.dataset import load_datasets
from utils.helpers import calculate_mean_std


def main(config_path):
    if not os.path.isabs(config_path):
        config_path = os.path.join(get_project_root(), config_path)

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    train_ds, _, _ = load_datasets(config, mean=None, std=None)
    mean, std = calculate_mean_std(train_ds)
    print("Training mean:", mean.tolist())
    print("Training std:", std.tolist())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/config.yaml')
    args = parser.parse_args()
    main(args.config)
