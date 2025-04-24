import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

# Create a figure with subplots
fig = plt.figure(figsize=(15, 10))

# Function to plot 3D surface


def plot_3d_surface(ax, X, W, Z, title):
    surf = ax.plot_surface(X, W, Z, linewidth=0, cmap='viridis')
    ax.set_xlabel('X')
    ax.set_ylabel('W')
    ax.set_zlabel('Error function')
    ax.set_title(title)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)


# 1. Full cost function visualization
ax1 = fig.add_subplot(221, projection='3d')
x = np.linspace(-1, 1, 50)
w = np.linspace(-1, 1, 50)
X, W = np.meshgrid(x, w)
Z = W*X**3 + X*W**2
plot_3d_surface(ax1, X, W, Z, 'Full Cost Function')


# 2. Fixed input x=-0.5
ax2 = fig.add_subplot(222, projection='3d')
x = np.linspace(-0.5, -0.51, 50)
w = np.linspace(-1, 1, 50)
X, W = np.meshgrid(x, w)
Z = W*X**3 + X*W**2
plot_3d_surface(ax2, X, W, Z, 'Fixed Input x=-0.5')

# 3. Fixed weight w=-1.0
ax3 = fig.add_subplot(223, projection='3d')
x = np.linspace(-1, 1, 50)
w = np.linspace(-0.5, -0.51, 50)
X, W = np.meshgrid(x, w)
Z = W*X**3 + X*W**2
plot_3d_surface(ax3, X, W, Z, 'Fixed Weight w=-1.0')

# 4. Few x cost function
ax4 = fig.add_subplot(224, projection='3d')
X = np.linspace(-5, 5, 10)
w = -1.0
W = np.linspace(w, w, 10)
x_s = []
w_s = []
z_s = []

for x in X:
    for w in W:
        z = w*x**3 + x*w**2
        z_s.append(z)
        x_s.append(x)
        w_s.append(w)

ax4.scatter(x_s, w_s, z_s, c='red', s=50)
ax4.set_xlabel('X')
ax4.set_ylabel('W')
ax4.set_zlabel('Error function')
ax4.set_title('Few x Cost Function')

plt.tight_layout()
plt.show()
