my_project/
├── config/
│   └── config.yaml                 # Training config (batch size, lr, etc.)
├── data/
│   ├── __init__.py
│   └── cifar10_loader.py          # Dataset and DataLoader setup
├── models/
│   ├── __init__.py
│   └── simple_cnn.py              # Model definition
├── train.py                       # Main training entry point
├── evaluate.py                    # Evaluation script (optional)
├── utils/
│   ├── __init__.py
│   └── misc.py                    # Logging, metrics, etc.
├── logs/                          # TensorBoard or wandb logs
├── checkpoints/                  # Saved model weights
├── requirements.txt              # Python dependencies
└── README.md

