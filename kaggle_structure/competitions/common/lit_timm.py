"""Reusable PyTorch Lightning + timm classifier.

Shared building block for the classification-style targets (B: APTOS DR, and any
medical classifier). Supports plain multi-class and *ordinal regression* (treat the
label as a scalar, round to the nearest class) which is what wins ordinal-metric
comps like APTOS (quadratic weighted kappa).

Example
-------
    from lit_timm import TimmClassifier
    model = TimmClassifier("tf_efficientnetv2_s.in21k_ft_in1k", num_classes=5, ordinal=True)
    trainer = pl.Trainer(precision="16-mixed", max_epochs=20)
    trainer.fit(model, datamodule)
"""
from __future__ import annotations

import timm
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.metrics import cohen_kappa_score


class TimmClassifier(pl.LightningModule):
    def __init__(
        self,
        backbone: str = "tf_efficientnetv2_s.in21k_ft_in1k",
        num_classes: int = 5,
        ordinal: bool = False,
        lr: float = 3e-4,
        weight_decay: float = 1e-5,
        pretrained: bool = True,
        drop_rate: float = 0.2,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters()
        # ordinal regression -> single scalar head; else one logit per class
        out_dim = 1 if ordinal else num_classes
        self.net = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=out_dim,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
        )
        self.ordinal = ordinal
        self.num_classes = num_classes
        self._val_preds: list[torch.Tensor] = []
        self._val_targets: list[torch.Tensor] = []

    def forward(self, x):
        return self.net(x)

    # ---- loss ----------------------------------------------------------------
    def _loss(self, logits, y):
        if self.ordinal:
            return F.mse_loss(logits.squeeze(1), y.float())
        return F.cross_entropy(logits, y)

    def _to_class(self, logits):
        if self.ordinal:
            return logits.squeeze(1).round().clamp(0, self.num_classes - 1).long()
        return logits.argmax(1)

    # ---- steps ---------------------------------------------------------------
    def training_step(self, batch, _):
        x, y = batch
        logits = self(x)
        loss = self._loss(logits, y)
        self.log("train/loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        logits = self(x)
        self.log("val/loss", self._loss(logits, y), prog_bar=True)
        self._val_preds.append(self._to_class(logits).cpu())
        self._val_targets.append(y.cpu())

    def on_validation_epoch_end(self):
        if not self._val_preds:
            return
        preds = torch.cat(self._val_preds).numpy()
        targets = torch.cat(self._val_targets).numpy()
        qwk = cohen_kappa_score(targets, preds, weights="quadratic")
        self.log("val/qwk", qwk, prog_bar=True)
        self._val_preds.clear()
        self._val_targets.clear()

    # ---- optim ---------------------------------------------------------------
    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.trainer.max_epochs or 20
        )
        return {"optimizer": opt, "lr_scheduler": sched}
