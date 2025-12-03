import torch
import matplotlib.pyplot as plt
import numpy as np

steps = 200                     # number of epochs or iterations to simulate
base_lr = 1e-3                  # starting LR


def new_opt(lr=base_lr):
    return torch.optim.Adam([torch.zeros(1)], lr=lr)


def record_scheduler(name, scheduler, optimizer, mode="epoch"):
    """
    mode = 'epoch' or 'iter'
    """
    lrs = []

    for t in range(steps):
        # required to avoid warning: optimizer.step() must run before scheduler.step()
        optimizer.step()

        if name == "ReduceLROnPlateau":
            # create fake validation loss curve
            # first improvements then plateau, then slight increase
            val_loss = 1 + np.sin(t / 15) * 0.1 + (t / steps) * 0.3
            scheduler.step(val_loss)
        else:
            scheduler.step()

        lrs.append(optimizer.param_groups[0]["lr"])

    return name, lrs


results = []

# ------------------------------------------------------------
# 1) StepLR
# ------------------------------------------------------------
opt = new_opt()
sched = torch.optim.lr_scheduler.StepLR(opt, step_size=40, gamma=0.5)
results.append(record_scheduler("StepLR", sched, opt))

# ------------------------------------------------------------
# 2) Cosine Annealing
# ------------------------------------------------------------
opt = new_opt()
sched = torch.optim.lr_scheduler.CosineAnnealingLR(
    opt, T_max=steps, eta_min=1e-6)
results.append(record_scheduler("CosineAnnealing", sched, opt))

# ------------------------------------------------------------
# 3) Warmup + Cosine (Linear warmup first 20 steps)
# ------------------------------------------------------------
total_warmup = 20
opt = new_opt()

scheduler_warmup = torch.optim.lr_scheduler.LinearLR(
    opt, start_factor=0.1, total_iters=total_warmup
)
scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
    opt, T_max=steps - total_warmup, eta_min=1e-6
)

lrs = []
for t in range(steps):
    opt.step()

    if t < total_warmup:
        scheduler_warmup.step()
    else:
        scheduler_cosine.step()

    lrs.append(opt.param_groups[0]["lr"])

results.append(("Warmup+Cosine", lrs))

# ------------------------------------------------------------
# 4) ReduceLROnPlateau
# ------------------------------------------------------------
opt = new_opt()
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
    opt, mode="min", factor=0.5, patience=10, threshold=1e-4
)
results.append(record_scheduler("ReduceLROnPlateau", sched, opt))

# ------------------------------------------------------------
# 5) ExponentialLR
# ------------------------------------------------------------
opt = new_opt()
sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.97)
results.append(record_scheduler("ExponentialLR", sched, opt))

# ------------------------------------------------------------
# 6) PolynomialLR
# ------------------------------------------------------------
opt = new_opt()
sched = torch.optim.lr_scheduler.PolynomialLR(opt, total_iters=steps, power=2)
results.append(record_scheduler("PolynomialLR", sched, opt))

# ------------------------------------------------------------
# 7) OneCycleLR (per iteration)
# ------------------------------------------------------------
max_lr = 1e-2
opt = new_opt(lr=1e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(
    opt, max_lr=max_lr, total_steps=steps
)
results.append(record_scheduler("OneCycleLR", sched, opt, mode="iter"))

# ------------------------------------------------------------
# 8) CyclicLR (per iteration)
# ------------------------------------------------------------
opt = new_opt(lr=1e-4)
sched = torch.optim.lr_scheduler.CyclicLR(
    opt,
    base_lr=1e-4,
    max_lr=1e-2,
    step_size_up=steps // 4,
    cycle_momentum=False,
)
results.append(record_scheduler("CyclicLR", sched, opt, mode="iter"))

# ------------------------------------------------------------
# 9) Linear Warmup only
# ------------------------------------------------------------
opt = new_opt(lr=1e-6)
sched = torch.optim.lr_scheduler.LinearLR(
    opt, start_factor=1e-3, total_iters=steps
)
results.append(record_scheduler("LinearWarmupOnly", sched, opt))

# ------------------------------------------------------------
# 10) LambdaLR
# ------------------------------------------------------------
opt = new_opt()


def lr_lambda(epoch):
    # simple example: inverse sqrt decay
    return 1.0 / np.sqrt(epoch + 1)


sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
results.append(record_scheduler("LambdaLR", sched, opt))

# ------------------------------------------------------------
# Plot all curves
# ------------------------------------------------------------
plt.figure(figsize=(14, 8))

for name, lrs in results:
    plt.plot(lrs, label=name)

plt.xlabel("Step")
plt.ylabel("Learning Rate")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
