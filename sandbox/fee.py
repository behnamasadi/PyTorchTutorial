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


def Sphere(x, y, x_0=2, y_0=2, z_0=4):
    return (x - x_0)**2 + (y - y_0)**2+z_0


# Initialize parameters for optimization
xy = torch.tensor([3.0, 3.0], requires_grad=True)  # Starting point
# optimizer = optim.Adam([xy], lr=0.1)
# optimizer = optim.SGD([xy], lr=0.1)
optimizer = optim.SGD([xy], lr=0.1, momentum=0.9)


scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)


path_x = []
path_y = []
path_z = []

x_start = xy[0].detach().item()
y_start = xy[1].detach().item()
# z_start = Rosenbrock(x_start, y_start)
z_start = Sphere(x_start, y_start)

# Optimization loop
n_iterations = 200
tolerance = 1e-3  # Tolerance for early stopping
min_grad_norm = 1e-3  # Minimum gradient norm for early stopping

for i in range(n_iterations):
    optimizer.zero_grad()
    loss = Sphere(xy[0], xy[1])
    loss.backward()

    # Calculate gradient norm for early stopping
    grad_norm = torch.norm(torch.tensor([xy.grad[0], xy.grad[1]]))

    optimizer.step()
    # scheduler.step()

    # Store the path
    path_x.append(xy[0].detach().item())
    path_y.append(xy[1].detach().item())
    path_z.append(loss.detach().item())

    if i % 10 == 0:
        print(
            f'Iteration {i}, Loss: {loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.6f}')
        print(
            f"xy: {xy.detach().numpy()}, loss: {Sphere(xy[0], xy[1]).item():.6f}")
        print(f"Gradient norm: {grad_norm.item():.6f}")

    # Early stopping conditions
    if loss.item() < tolerance:
        print(f"Early stopping: Loss below tolerance {tolerance}")
        break
    if grad_norm < min_grad_norm:
        print(f"Early stopping: Gradient norm below threshold {min_grad_norm}")
        break


# Create a grid of points
x = torch.linspace(-1, 5, 20)  # Adjusted range to show the minimum at (2,2)
y = torch.linspace(-1, 5, 20)
X, Y = torch.meshgrid(x, y, indexing='ij')

# Calculate the function values
Z = Sphere(X, Y)

# Calculate partial derivatives
X.requires_grad_(True)
Y.requires_grad_(True)
Z = Sphere(X, Y)

# Calculate gradients
grad_x = torch.autograd.grad(Z.sum(), X, create_graph=True)[0]
grad_y = torch.autograd.grad(Z.sum(), Y, create_graph=True)[0]

# Calculate gradient magnitudes for scaling
grad_magnitude = torch.sqrt(grad_x**2 + grad_y**2)
max_magnitude = grad_magnitude.max().item()
# Reduced scale factor for better visualization
scale_factor = 0.3 / max_magnitude

# Create the figure
fig = plt.figure(figsize=(12, 10))
axe = fig.add_subplot(111, projection='3d')

# Plot the surface
axe.plot_surface(X.detach().numpy(), Y.detach().numpy(), Z.detach().numpy(),
                 cmap=cm.coolwarm, alpha=0.5)

# Create a 2D quiver plot of the gradient field with proportional sizes
axe.quiver(X.detach().numpy(), Y.detach().numpy(), Z.detach().numpy(),
           grad_x.detach().numpy(), grad_y.detach().numpy(),
           torch.zeros_like(Z).detach().numpy(),
           length=scale_factor, normalize=False, color='black')

# Add labels and title
axe.set_xlabel('X')
axe.set_ylabel('Y')
axe.set_zlabel('Z')
axe.set_title(
    'Shifted Sphere Function with Gradient Field (Proportional Magnitude)')

axe.scatter(x_start, y_start, z_start, color='green', marker='x')
axe.plot(path_x, path_y, path_z, 'r-')

print("x,y,z", xy[0].detach().item(),
      xy[1].detach().item(), Sphere(xy[0], xy[1]).item())


plt.show()

plt.plot(path_z)
plt.title("Loss over iterations")
plt.show()
# exit()
