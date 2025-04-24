import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import cm
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
import torch.utils.data.dataset as dataset


def ackley_function(x, y):
    """
    Calculates the value of the Ackley function at the point (x, y).

    The Ackley function is a non-convex function, often used as a
    performance test problem for optimization algorithms.

    Args:
      x: The x-coordinate of the point.
      y: The y-coordinate of the point.

    Returns:
      The value of the Ackley function at (x, y).
    """
    return -20*np.exp(-0.2*np.sqrt(0.5*(x**2 + y**2))) - np.exp(0.5*(np.cos(2*np.pi*x) + np.cos(2*np.pi*y))) + 20 + np.e


g = torch.Generator()
low = -5
high = 5
g.manual_seed(42)
points_2d = torch.rand(size=(2000, 2), generator=g) * (high - low) + low
print("Generated 2D points:\n", points_2d)

z = ackley_function(points_2d[:, 0], points_2d[:, 1])
print("Ackley function values (z):\n", z)

# Convert PyTorch tensors to NumPy arrays for plotting
x_np = points_2d[:, 0].numpy()
y_np = points_2d[:, 1].numpy()
z_np = z.numpy()

# Create the 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Tri-Surface plot of the points
ax.plot_trisurf(x_np, y_np, z_np, cmap=cm.viridis,
                linewidth=0.2, antialiased=True)

# Add labels and title
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z (Ackley Function Value)')
ax.set_title('3D Surface Plot of Ackley Function Values')

# Show the plot
plt.show()


class functionApproximation(nn.Module):
    def __init__(self):
        super().__init__(input_dim=2, output_dim=1, hidden_dim=100)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = functionApproximation()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
