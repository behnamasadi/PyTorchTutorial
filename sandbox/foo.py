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


# Create a grid of points (using fewer points for 3D visualization)
# x = torch.linspace(-10, 10, 200)  # Reduced from 50 to 20 for clearer 3D plot
# y = torch.linspace(-10, 10, 200)

x = torch.linspace(-5, 5, 200)  # Reduced from 50 to 20 for clearer 3D plot
y = torch.linspace(-5, 5, 200)


X, Y = torch.meshgrid(x, y, indexing='ij')

# Z = Ackley(X, Y)
Z = Rosenbrock(X, Y)

fix, axe = plt.subplots(nrows=1, ncols=1, figsize=(
    10, 10), subplot_kw={'projection': '3d'})

axe.plot_surface(X.detach().numpy(), Y.detach().numpy(),
                 Z.detach().numpy(), cmap=cm.coolwarm, alpha=0.5)

plt.show()


# exit()


# Calculate Z values for the surface
Z = torch.zeros_like(X)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Z[i, j] = Rosenbrock(X[i, j], Y[i, j])

# Convert tensors to lists for plotting
X_list = [[float(X[i, j]) for j in range(X.shape[1])]
          for i in range(X.shape[0])]
Y_list = [[float(Y[i, j]) for j in range(Y.shape[1])]
          for i in range(Y.shape[0])]
Z_list = [[float(Z[i, j]) for j in range(Z.shape[1])]
          for i in range(Z.shape[0])]

# Initialize parameters for optimization
xy = torch.tensor([3.0, 3.0], requires_grad=True)  # Starting point
optimizer = optim.Adam([xy], lr=0.1)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

# Store optimization path
path_x = []
path_y = []
path_z = []

# Store initial point
path_x.append(xy[0].item())
path_y.append(xy[1].item())
path_z.append(Rosenbrock(xy[0], xy[1]).item())

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

# Create figure with subplots
plt.figure(figsize=(18, 6))

# 3D Surface Plot
ax1 = plt.subplot(131, projection='3d')

# Plot 3D lines
for i in range(len(X_list)):
    ax1.plot([X_list[i][j] for j in range(len(X_list[i]))],
             [Y_list[i][j] for j in range(len(Y_list[i]))],
             [Z_list[i][j] for j in range(len(Z_list[i]))],
             'b-', alpha=0.2)

for j in range(len(X_list[0])):
    ax1.plot([X_list[i][j] for i in range(len(X_list))],
             [Y_list[i][j] for i in range(len(Y_list))],
             [Z_list[i][j] for i in range(len(Z_list))],
             'b-', alpha=0.2)

# Plot optimization path
ax1.plot(path_x, path_y, path_z, 'r.-', linewidth=2, label='Optimization Path')
ax1.scatter([path_x[0]], [path_y[0]], [path_z[0]],
            color='green', s=100, label='Start')
ax1.scatter([path_x[-1]], [path_y[-1]], [path_z[-1]],
            color='red', s=100, label='End')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_title('3D Rosenbrock Function')
ax1.legend()

# Contour Plot
ax2 = plt.subplot(132)
contour = ax2.contour(X_list, Y_list, Z_list, levels=50, cmap=cm.coolwarm)
plt.colorbar(contour, label='Rosenbrock Function Value')
ax2.plot(path_x, path_y, 'k.-', linewidth=2, label='Optimization Path')
ax2.scatter([path_x[0]], [path_y[0]], color='green', s=100, label='Start')
ax2.scatter([path_x[-1]], [path_y[-1]], color='red', s=100, label='End')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_title('Contour Plot')
ax2.grid(True)
ax2.legend()

# Convergence Plot
ax3 = plt.subplot(133)
ax3.plot(path_z, 'b-', label='Loss Value')
ax3.set_xlabel('Iteration')
ax3.set_ylabel('Loss')
ax3.set_title('Convergence Plot')
ax3.grid(True)
ax3.legend()

plt.tight_layout()
plt.show()

print(f'\nFinal position: x={path_x[-1]:.4f}, y={path_y[-1]:.4f}')
print(f'Final value: {path_z[-1]:.4f}')
