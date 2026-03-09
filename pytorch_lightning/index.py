"""
PyTorch Lightning: Function approximation example.
Approximates y = x^2 * sin(x) with a 4-layer MLP.

Before training (optional, for live dashboards):
  TensorBoard:  tensorboard --logdir logs
  MLflow:       mlflow server --host 127.0.0.1 --port 5000 \\
                --backend-store-uri sqlite:///mlflow.db \\
                --artifacts-destination ./artifacts
"""

from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger, WandbLogger, MLFlowLogger
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
    DeviceStatsMonitor,
    RichProgressBar,
    ModelSummary,
    GradientAccumulationScheduler,
    ModelPruning,
)
from pytorch_lightning.tuner import Tuner
from pytorch_lightning.callbacks import Callback
import pytorch_lightning as pl
from torch.nn import MSELoss
from torch.utils.data import TensorDataset, DataLoader, random_split
import torch
import warnings
warnings.filterwarnings("ignore")


class DebugCallback(Callback):
    """Minimal debug callback: prints at train start/end."""

    def on_train_start(self, trainer, pl_module):
        print("Training started")

    def on_train_end(self, trainer, pl_module):
        print("Training ended")


def func(x):
    """Target function: y = x^2 * sin(x)."""
    return (x**2) * torch.sin(x)


class FuncApproximate(torch.nn.Module):
    """MLP: 1 -> 64 -> 256 -> 64 -> 1 with LeakyReLU."""

    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(1, 64)
        self.fc2 = torch.nn.Linear(64, 256)
        self.fc3 = torch.nn.Linear(256, 64)
        self.fc4 = torch.nn.Linear(64, 1)
        self.relu = torch.nn.LeakyReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return self.fc4(x)


class LitModelFuncApproximate(pl.LightningModule):
    """LightningModule with AdamW, ExponentialLR, train/val/test steps."""

    def __init__(self, model, loss_fn, lr=1e-4, weight_decay=0.01, scheduler_gamma=0.9):
        super().__init__()
        self.save_hyperparameters(ignore=["model", "loss_fn"])
        self.model = model
        self.loss_fn = loss_fn

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss_fn(self.model(x), y)
        self.log("train_loss", loss, on_step=False,
                 on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss_fn(self(x), y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        """Print train/val loss at end of each epoch."""
        train_loss = self.trainer.callback_metrics.get("train_loss")
        val_loss = self.trainer.callback_metrics.get("val_loss")
        t = f"{float(train_loss):.4f}" if train_loss is not None else "N/A"
        v = f"{float(val_loss):.4f}" if val_loss is not None else "N/A"
        print(f"Epoch {self.current_epoch} - train_loss: {t}, val_loss: {v}")

    def test_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss_fn(self(x), y)
        self.log("test_loss", loss, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        # Optimizer: AdamW with lr, weight_decay from hparams
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        # Scheduler: ExponentialLR decays lr by gamma each epoch
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            opt, gamma=self.hparams.scheduler_gamma
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}


class LitModelDataFuncApproximate(pl.LightningDataModule):
    """DataModule for pre-split train/val/test datasets."""

    def __init__(self, train_dataset, val_dataset, test_dataset, batch_size=64, num_workers=4):
        super().__init__()
        self.save_hyperparameters(
            ignore=["train_dataset", "val_dataset", "test_dataset"])
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            pin_memory=True,
            num_workers=self.hparams.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            pin_memory=True,
            num_workers=self.hparams.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            pin_memory=True,
            num_workers=self.hparams.num_workers,
        )


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("medium")

    # Data generation
    num_sample = 10000
    x = torch.linspace(-4, 4, num_sample, dtype=torch.float32).unsqueeze(-1)
    y = func(x)

    x_min, x_max = -4.0, 4.0
    x_norm = 2.0 * (x - x_min) / (x_max - x_min) - 1.0
    y_norm = y / y.abs().max()

    dataset = TensorDataset(x_norm, y_norm)
    train_size = int(0.7 * num_sample)
    val_size = int(0.15 * num_sample)
    test_size = num_sample - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        lengths=[train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"Train: {train_size}, Val: {val_size}, Test: {test_size}")

    # DataModule and model
    data_module = LitModelDataFuncApproximate(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=64,
        num_workers=4,
    )

    lit_model = LitModelFuncApproximate(
        model=FuncApproximate(),
        loss_fn=MSELoss(reduction="mean"),
        lr=1e-4,
        weight_decay=0.01,
        scheduler_gamma=0.9,
    )

    # --- Stage 1: Tune batch size (Lightning 2.x: use Tuner.scale_batch_size) ---
    tune_trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=1,
    )
    Tuner(tune_trainer).scale_batch_size(
        lit_model,
        datamodule=data_module,
        mode="power",
        max_val=256,
    )
    print(f"Stage 1: Suggested batch_size = {data_module.hparams.batch_size}")

    # --- Stage 2: Full training with gradient accumulation + pruning ---
    lit_model_stage2 = LitModelFuncApproximate(
        model=FuncApproximate(),
        loss_fn=MSELoss(reduction="mean"),
        lr=1e-4,
        weight_decay=0.01,
        scheduler_gamma=0.9,
    )

    checkpoint = ModelCheckpoint(
        dirpath="checkpoints/func_approximate",
        filename="best-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )
    early_stopping = EarlyStopping(monitor="val_loss", patience=20, mode="min")
    learning_rate_monitor = LearningRateMonitor(logging_interval="epoch")
    device_stats_monitor = DeviceStatsMonitor()
    model_summary = ModelSummary(max_depth=2)
    gradient_accumulation_scheduler = GradientAccumulationScheduler({0: 1, 5: 4})
    model_pruning = ModelPruning(pruning_fn="l1_unstructured", amount=0.3)

    callbacks = [
        DebugCallback(),
        checkpoint,
        early_stopping,
        learning_rate_monitor,
        device_stats_monitor,
        model_summary,
        gradient_accumulation_scheduler,
        model_pruning,
        RichProgressBar(),
    ]

    loggers = [
        CSVLogger(save_dir="logs", name="func_approximate"),
        TensorBoardLogger(save_dir="logs", name="func_approximate"),
        MLFlowLogger(
            experiment_name="func_approximate",
            tracking_uri="http://127.0.0.1:5000",
        ),
    ]

    trainer = pl.Trainer(
        max_epochs=200,
        accelerator="auto",
        devices=1,
        precision="16-mixed",
        log_every_n_steps=10,
        callbacks=callbacks,
        logger=loggers,
    )

    trainer.fit(lit_model_stage2, datamodule=data_module)
    trainer.test(lit_model_stage2, datamodule=data_module)


if __name__ == "__main__":
    main()
