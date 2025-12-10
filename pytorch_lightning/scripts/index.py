import time
import pytorch_lightning as pl
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, Dataset, random_split
import matplotlib.pyplot as plt
import os
import json
from pytorch_lightning.callbacks import BatchSizeFinder, BackboneFinetuning, ModelCheckpoint, DeviceStatsMonitor, EarlyStopping, GradientAccumulationScheduler, LearningRateMonitor, ModelPruning, ModelSummary, Callback
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger, MLFlowLogger, NeptuneLogger
from torchvision.transforms import Compose

# #region agent log
DEBUG_LOG_PATH = "/home/behnam/anaconda3/envs/PyTorchTutorial/.cursor/debug.log"


def _debug_log(session_id, run_id, hypothesis_id, location, message, data):
    try:
        os.makedirs(os.path.dirname(DEBUG_LOG_PATH), exist_ok=True)
        with open(DEBUG_LOG_PATH, "a") as f:
            f.write(json.dumps({"sessionId": session_id, "runId": run_id, "hypothesisId": hypothesis_id,
                    "location": location, "message": message, "data": data, "timestamp": time.time() * 1000}) + "\n")
    except Exception as e:
        # Fallback to stderr if file logging fails
        import sys
        print(f"DEBUG_LOG_ERROR: {e}", file=sys.stderr)
# #endregion


class Normalize:
    """Normalize 1D tensor using mean and std."""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        return (x - self.mean) / self.std


class FunctionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)

        x = torch.relu(self.fc2(x))
        x = self.dropout(x)

        x = torch.relu(self.fc3(x))

        x = self.fc4(x)
        return x


class FunctionDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class TransformDataset(Dataset):
    """Wrapper to apply transforms to a dataset."""

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        if self.transform is not None:
            x = self.transform(x)
        return x, y


class LitDataModule(pl.LightningDataModule):
    def __init__(self, dataset, batch_size=32, train_split=0.8, val_split=0.1, num_workers=0, transform=None):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.train_split = train_split
        self.val_split = val_split
        self.num_workers = num_workers
        self.transform = transform
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            # Calculate split sizes
            total_size = len(self.dataset)
            train_size = int(self.train_split * total_size)
            val_size = int(self.val_split * total_size)
            test_size = total_size - train_size - val_size

            # Split dataset
            train_subset, val_subset, test_subset = random_split(
                self.dataset, [train_size, val_size, test_size]
            )

            # Apply transforms
            self.train_ds = TransformDataset(
                train_subset, transform=self.transform)
            self.val_ds = TransformDataset(
                val_subset, transform=self.transform)
            self.test_ds = TransformDataset(
                test_subset, transform=self.transform)

        if stage == "test" or stage is None:
            if self.test_ds is None:
                # If test split wasn't created during fit, create it now
                total_size = len(self.dataset)
                train_size = int(self.train_split * total_size)
                val_size = int(self.val_split * total_size)
                test_size = total_size - train_size - val_size
                _, _, test_subset = random_split(
                    self.dataset, [train_size, val_size, test_size]
                )
                self.test_ds = TransformDataset(
                    test_subset, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True  # Speeds up GPU transfer
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True  # Speeds up GPU transfer
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True  # Speeds up GPU transfer
        )


class DebugCallback(Callback):
    """Callback to track execution order and pruning events"""

    def on_train_epoch_end(self, trainer, pl_module):
        # #region agent log
        _debug_log("debug-session", "run1", "C", "index.py:on_train_epoch_end", "train_epoch_end",
                   {"epoch": trainer.current_epoch, "val_loss": trainer.callback_metrics.get("val_loss", None)})
        # #endregion

    def on_validation_epoch_end(self, trainer, pl_module):
        # #region agent log
        _debug_log("debug-session", "run1", "B", "index.py:on_validation_epoch_end", "validation_epoch_end", {"epoch": trainer.current_epoch, "val_loss": float(
            trainer.callback_metrics.get("val_loss", float("nan")).item()) if trainer.callback_metrics.get("val_loss") is not None else None})
        # #endregion

    def on_train_start(self, trainer, pl_module):
        # #region agent log
        _debug_log("debug-session", "run1", "A", "index.py:on_train_start",
                   "on_train_start", {"accumulate_grad_batches": trainer.accumulate_grad_batches, "has_grad_scheduler": any(isinstance(cb, GradientAccumulationScheduler) for cb in trainer.callbacks)})
        # #endregion

    def on_exception(self, trainer, pl_module, exception):
        # #region agent log
        _debug_log("debug-session", "run1", "ALL", "index.py:on_exception", "exception_in_callback",
                   {"error": str(exception), "error_type": type(exception).__name__})
        # #endregion


class LitModel(pl.LightningModule):
    def __init__(self, model, criterion, lr=1e-3):
        super().__init__()
        self.model = model
        self.loss_fn = criterion
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.loss_fn(pred, y)
        self.log("training_loss", loss, prog_bar=True,
                 on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.loss_fn(preds, y)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        # #region agent log
        _debug_log("debug-session", "run1", "B", "index.py:181", "validation_step",
                   {"batch_idx": batch_idx, "loss": float(loss.item()), "epoch": self.current_epoch})
        # #endregion
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.loss_fn(preds, y)
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== CONFIGURATION: Change these values to test different ranges =====
x_min, x_max = -20, 20  # Try: (-10, 10), (0, 50), (-100, 100), etc.
n_samples = 4000
# ========================================================================

# Generate data (keep on CPU for DataLoader workers, will move to GPU in training step)
x = torch.linspace(x_min, x_max, n_samples).reshape(-1, 1)
y = torch.sin(x) + 0.05 * torch.randn(n_samples, 1)

# Convert to numpy for plotting
x_np = x.cpu().numpy()
y_np = y.cpu().numpy()

plt.plot(x_np, y_np, ".")
plt.title(f"Generated Training Data (x range: [{x_min}, {x_max}])")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# Calculate normalization parameters
x_mean = x.mean()
x_std = x.std()
y_mean = y.mean()
y_std = y.std()

# Store normalization parameters for later denormalization
norm_params = {
    'x_mean': x_mean,
    'x_std': x_std,
    'y_mean': y_mean,
    'y_std': y_std,
    'x_min': x_min,
    'x_max': x_max
}

print(f"Original x range: [{x.min():.2f}, {x.max():.2f}]")
print(f"Original x mean: {x_mean:.4f}, std: {x_std:.4f}")
print(f"Original y range: [{y.min():.2f}, {y.max():.2f}]")
print(f"Original y mean: {y_mean:.4f}, std: {y_std:.4f}")

# Define PyTorch transform: normalize with mean/std
transform = Compose([
    Normalize(mean=x_mean, std=x_std)
])

# Create dataset
dataset = FunctionDataset(x, y)

# Create data module with transform
# Calculate max batch size (should be less than dataset size)
train_size = int(0.8 * len(dataset))
# Cap at 512 or dataset size, whichever is smaller
max_batch_size = min(512, train_size)

# Calculate num_workers: CPU count - 1, with protection to ensure >= 1
num_workers = max(1, os.cpu_count() - 1) if os.cpu_count() else 1

print("CPU count: ", os.cpu_count())


data_module = LitDataModule(
    dataset=dataset, batch_size=32, transform=transform, num_workers=num_workers)

# Create model
model = LitModel(model=FunctionModel(),
                 criterion=torch.nn.MSELoss(), lr=1e-3)

# Callbacks
# #region agent log
_debug_log("debug-session", "run1", "A", "index.py:255", "Creating callbacks",
           {"checkpoint_monitor": "val_loss", "early_stopping_patience": 5, "pruning_amount": 0.5, "grad_accum_schedule": {0: 1, 5: 4}})
# #endregion
checkpoint = ModelCheckpoint(
    monitor="val_loss",
    save_top_k=3,
    mode="min"
)

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    mode="min"
)

learning_rate_monitor = LearningRateMonitor(logging_interval="epoch")

device_stats_monitor = DeviceStatsMonitor()

model_summary = ModelSummary(max_depth=2)

gradient_accumulation_scheduler = GradientAccumulationScheduler({
    0: 1,  # epoch 0: accumulate 1 batch
    5: 4,  # epoch 5+: accumulate 4 batches
})

# ModelPruning callback (example configuration)
model_pruning = ModelPruning(
    pruning_fn="l1_unstructured",
    amount=0.5
)
# #region agent log
_debug_log("debug-session", "run1", "A", "index.py:282",
           "Callbacks created successfully", {"callback_count": 7})
# #endregion

# NOTE: BatchSizeFinder should NOT be used with GradientAccumulationScheduler or ModelPruning
# It should be run separately before training. See below for optional batch size tuning.

# Create trainer with safe callbacks only
# #region agent log
try:
    wandb_logger = WandbLogger(
        project="function_approximation_pytorch_lightning")
    _debug_log("debug-session", "run1", "D", "index.py:292", "WandbLogger initialized",
               {"project": "function_approximation_pytorch_lightning", "success": True})
except Exception as e:
    _debug_log("debug-session", "run1", "D", "index.py:292", "WandbLogger initialization failed",
               {"error": str(e), "error_type": type(e).__name__, "success": False})
    wandb_logger = None
# #endregion
# #region agent log
_debug_log("debug-session", "run1", "A", "index.py:356",
           "Creating Trainer (before)", {"will_use_grad_scheduler": True})
# #endregion
# NOTE: Do NOT set accumulate_grad_batches when using GradientAccumulationScheduler
# The callback requires the default value (1) and will manage it from there
trainer = pl.Trainer(
    accelerator="gpu",
    devices=1,
    precision="16-mixed",
    logger=wandb_logger if wandb_logger else None,
    max_epochs=100,
    log_every_n_steps=10,  # Lower for small datasets
    # accumulate_grad_batches is NOT set - defaults to 1, which GradientAccumulationScheduler requires
    callbacks=[
        DebugCallback(),  # Track callback execution order
        checkpoint,
        early_stopping,
        learning_rate_monitor,
        device_stats_monitor,
        model_summary,
        gradient_accumulation_scheduler,  # Handles gradient accumulation scheduling
        model_pruning
    ]
)
# #region agent log
_debug_log("debug-session", "run1", "A", "index.py:377", "Trainer created (after)",
           {"accumulate_grad_batches": trainer.accumulate_grad_batches, "type": type(trainer.accumulate_grad_batches).__name__, "has_grad_scheduler": True, "has_pruning": True})
# #endregion

# Optional: Run batch size finder separately BEFORE training
# Uncomment the following to find optimal batch size (run this once, then remove):
# train_size = int(0.8 * len(dataset))
# max_batch_size = min(512, train_size)
# batch_size_finder = BatchSizeFinder(mode="power", max_val=max_batch_size)
# tune_trainer = pl.Trainer(
#     accelerator="gpu",
#     devices=1,
#     max_epochs=1,
#     callbacks=[batch_size_finder]
# )
# tune_trainer.tune(model, datamodule=data_module)
# # After tuning, update batch_size in LitDataModule and remove BatchSizeFinder

# Train the model
# #region agent log
_debug_log("debug-session", "run1", "ALL", "index.py:321", "Starting trainer.fit()",
           {"model_params": sum(p.numel() for p in model.parameters()), "dataset_size": len(dataset)})
# #endregion
try:
    trainer.fit(model, data_module)
    # #region agent log
    _debug_log("debug-session", "run1", "ALL", "index.py:321", "trainer.fit() completed successfully",
               {"current_epoch": trainer.current_epoch, "global_step": trainer.global_step})
    # #endregion
except Exception as e:
    # #region agent log
    _debug_log("debug-session", "run1", "ALL", "index.py:321", "trainer.fit() failed",
               {"error": str(e), "error_type": type(e).__name__, "traceback": __import__("traceback").format_exc()})
    # #endregion
    raise
# #region agent log
_debug_log("debug-session", "run1", "ALL",
           "index.py:322", "Starting trainer.test()", {})
# #endregion
try:
    trainer.test(model, data_module)
    # #region agent log
    _debug_log("debug-session", "run1", "ALL", "index.py:322",
               "trainer.test() completed successfully", {})
    # #endregion
except Exception as e:
    # #region agent log
    _debug_log("debug-session", "run1", "ALL", "index.py:322",
               "trainer.test() failed", {"error": str(e), "error_type": type(e).__name__})
    # #endregion
    raise
