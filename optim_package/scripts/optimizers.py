import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

# 1. Full cost function visualization
x = np.linspace(-1, 1, 50)
w = np.linspace(-1, 1, 50)
X, W = np.meshgrid(x, w)
Z = W*X**3 + X*W**2

fig1 = plt.figure(figsize=(10, 8))
ax1 = fig1.add_subplot(111, projection='3d')
surf1 = ax1.plot_surface(X, W, Z, linewidth=0, cmap='viridis')
ax1.set_xlabel('X')
ax1.set_ylabel('W')
ax1.set_zlabel('Error function')
ax1.set_title('Full Cost Function')
fig1.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)
plt.show()

# 2. Fixed input x=-0.5
x = np.linspace(-0.5, -0.51, 50)
w = np.linspace(-1, 1, 50)
X, W = np.meshgrid(x, w)
Z = W*X**3 + X*W**2

fig2 = plt.figure(figsize=(10, 8))
ax2 = fig2.add_subplot(111, projection='3d')
surf2 = ax2.plot_surface(X, W, Z, linewidth=0, cmap='viridis')
ax2.set_xlabel('X')
ax2.set_ylabel('W')
ax2.set_zlabel('Error function')
ax2.set_title('Fixed Input x=-0.5')
fig2.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)
plt.show()

# 3. Fixed weight w=-1.0
x = np.linspace(-1, 1, 50)
w = np.linspace(-0.5, -0.51, 50)
X, W = np.meshgrid(x, w)
Z = W*X**3 + X*W**2

fig3 = plt.figure(figsize=(10, 8))
ax3 = fig3.add_subplot(111, projection='3d')
surf3 = ax3.plot_surface(X, W, Z, linewidth=0, cmap='viridis')
ax3.set_xlabel('X')
ax3.set_ylabel('W')
ax3.set_zlabel('Error function')
ax3.set_title('Fixed Weight w=-1.0')
fig3.colorbar(surf3, ax=ax3, shrink=0.5, aspect=5)
plt.show()

# 4. Few x cost function
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

fig4 = plt.figure(figsize=(10, 8))
ax4 = fig4.add_subplot(111, projection='3d')
ax4.scatter(x_s, w_s, z_s, c='red', s=50)
ax4.set_xlabel('X')
ax4.set_ylabel('W')
ax4.set_zlabel('Error function')
ax4.set_title('Few x Cost Function')
plt.show()

# Rosenbrock function (commented out for now)
# f(x,y) = (a-x)**2 + b*(y-x**2)**2
