import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting
import torch
import torch.optim as optim
import numpy as np
# Rosenbrock function


def Rosenbrock(x, y):
    return (1 - x)**2 + 100*(y - x**2)**2


def Ackley(x, y):
    return -20*torch.exp(-0.2*torch.sqrt(0.5*(x**2 + y**2))) - \
        torch.exp(0.5*(torch.cos(2*torch.pi*x) + torch.cos(2*torch.pi*y))) + \
        torch.exp(torch.tensor(1.0)) + 20


# Initialize parameters for optimization
xy = torch.tensor([3.0, 3.0], requires_grad=True)  # Starting point
optimizer = optim.Adam([xy], lr=0.1)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)


path_x = []
path_y = []
path_z = []

x_start = xy[0].detach().item()
y_start = xy[1].detach().item()
z_start = Rosenbrock(x_start, y_start)

# Optimization loop
n_iterations = 100
for i in range(n_iterations):
    optimizer.zero_grad()
    loss = Rosenbrock(xy[0], xy[1])
    loss.backward()
    optimizer.step()
    scheduler.step()

    # Store the path
    path_x.append(xy[0].detach().item())
    path_y.append(xy[1].detach().item())
    path_z.append(loss.detach().item())

    if i % 10 == 0:
        print(
            f'Iteration {i}, Loss: {loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.6f}')


# Create a grid of points (using fewer points for 3D visualization)
x = torch.linspace(-5, 5, 200)  # Reduced from 50 to 20 for clearer 3D plot
y = torch.linspace(-5, 5, 200)


X, Y = torch.meshgrid(x, y, indexing='ij')

# Z = Ackley(X, Y)
Z = Rosenbrock(X, Y)

fix, axe = plt.subplots(nrows=1, ncols=1, figsize=(
    10, 10), subplot_kw={'projection': '3d'})

axe.plot_surface(X.detach().numpy(), Y.detach().numpy(),
                 Z.detach().numpy(), cmap=cm.coolwarm, alpha=0.5)


axe.scatter(x_start, y_start, z_start, color='green', marker='x')
axe.plot(path_x, path_y, path_z, 'r-')


plt.show()


# exit()
