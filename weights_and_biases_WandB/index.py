import random
import torch
import torch.nn as nn
import wandb
import numpy as np
import os
# import os
# os.environ["WANDB_MODE"] = "online"
print(os.environ.get("WANDB_DIR"))
wandb.login()
wandb.init(project="my-project",
           config={"learning_rate": 0.01, "epochs": 5, "batch_size": 64})


for epoch in range(4):
    print(f"---------------- epoch: {epoch} ----------------")

    # random loss
    loss = np.random.randn(1)
    loss = loss.item()    # make it a scalar float
    wandb.log({"loss": loss})

    # random weights
    weights = np.random.randn(1000)

    # prepare everything in one dictionary
    metrics = {
        "epoch": epoch,      # good idea to log epoch explicitly
        "loss": loss
    }

    for name in range(3):
        metrics[f"weights/{name}"] = wandb.Histogram(weights)

    # now only one log call per epoch
    wandb.log(metrics)
    for k, v in metrics.items():
        print(k, v)

wandb.finish()
