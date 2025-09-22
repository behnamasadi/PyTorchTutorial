import wandb
import os
import math

# wandb.require("core")
wandb.login()
project = "simulated-experiment"
config = {
    "lr": 0.001,
    "model": "CNN",
    "weight": True
}
with wandb.init(project=project, config=config, name="") as run:
    epochs = 10
    for epoch in range(1, epochs):
        loss = 1/(epoch)
        acc = 1 - 2/(epoch*epoch)

        run.log({"acc": acc, "loss": loss})

cfg = wandb.config
print(cfg)
