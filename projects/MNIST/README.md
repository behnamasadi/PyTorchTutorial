# MNIST Classification Project

This project implements a Multi-Layer Perceptron (MLP) for classifying the MNIST dataset.

## Project Structure

```
your_project/
│
├── configs/                   # YAML or JSON config files for experiments
│   └── default.yaml
│
├── data/                      # Custom datasets and data loaders
│   ├── __init__.py
│   └── mnist_dataset.py
│
├── models/                    # Model architectures
│   ├── __init__.py
│   └── mlp.py
│
├── trainers/                  # Training loops, validation, and testing logic
│   ├── __init__.py
│   └── base_trainer.py
│
├── utils/                     # Utility functions (e.g., metrics, logging)
│   ├── __init__.py
│   ├── metrics.py
│   └── logger.py
│
├── experiments/               # Saved experiment results
│
├── main.py                    # Entry point to start training
├── evaluate.py                # Script to evaluate a saved model
├── requirements.txt           # Python dependencies
└── README.md
```

## Setup

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Make sure you have PyTorch and torchvision installed.

## Training

To train the model with the default configuration:

```bash
python main.py
```

This will:
1. Load the configuration from `configs/default.yaml`
2. Train the model using the specified hyperparameters
3. Save the best model, hyperparameters, and metrics to the `experiments` directory
4. Print the final test accuracy

## Evaluation

To evaluate a saved model:

```bash
python evaluate.py --experiment_dir experiments/experiment_TIMESTAMP
```

Replace `TIMESTAMP` with the actual timestamp of the experiment you want to evaluate.

This will:
1. Load the saved model and hyperparameters
2. Evaluate the model on the test set
3. Print the test accuracy and hyperparameters

## Reproducibility

The project is designed for reproducibility:
- Random seeds are set for all random operations
- All hyperparameters are saved in the experiment directory
- The configuration file is saved with the experiment
- The best model is saved during training

To reproduce an experiment:
1. Load the saved configuration and model
2. Set the same random seed
3. Train the model with the same hyperparameters

## Customization

You can modify the hyperparameters in `configs/default.yaml` to experiment with different configurations.
