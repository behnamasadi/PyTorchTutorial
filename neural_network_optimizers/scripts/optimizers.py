import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm


########################################### cost function #####################################################

x = np.linspace(-1,1,50)
w = np.linspace(-1,1,50)

X, W = np.meshgrid(x,w)
Z=W*X**3+X*W**2

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, W, Z,linewidth=0,cmap='viridis')
ax.set_xlabel('X')
ax.set_ylabel('W')
ax.set_zlabel('Error function')

plt.show()

############################################# Fixed input x=-0.5 ###################################################

x = np.linspace(-0.5,-0.51,50)
w = np.linspace(-1,1,50)

X, W = np.meshgrid(x,w)
Z=W*X**3+X*W**2


fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, W, Z,linewidth=0,cmap='viridis')
ax.set_xlabel('X')
ax.set_ylabel('W')
ax.set_zlabel('Error function')
plt.show()




############################################ Fixed weight w=-1.0####################################################
x = np.linspace(-1,1,50)
w = np.linspace(-0.5,-0.51,50)

X, W = np.meshgrid(x,w)
Z=W*X**3+X*W**2

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, W, Z,linewidth=0,cmap='viridis')
ax.set_xlabel('X')
ax.set_ylabel('W')
ax.set_zlabel('Error function')
plt.show()


####################################### few x cost function #############################################
fig = plt.figure()
ax = fig.gca(projection='3d')

ax.set_xlabel('X')
ax.set_ylabel('W')
ax.set_zlabel('Error function')


X=np.linspace(-5,5,10)
w=-1.0
#w=-0.0
W=np.linspace(w,w,10)

x_s=[]
w_s=[]
z_s=[]

for x in X:
        for w in W:
                z=w*x**3+x*w**2
                z_s.append(z)
                x_s.append(x)
                w_s.append(w)
                # print(x)

ax.scatter(x_s, w_s, z_s)
plt.show()


####################################### Rosenbrock function #############################################
# is a non-convex
# f(x,y)=(a-x)**2+b*(y-x**2)**2


