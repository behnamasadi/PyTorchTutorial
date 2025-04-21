import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm



x = np.linspace(-20,20,100)
y = np.linspace(-50,50,100)
X, Y = np.meshgrid(x,y)
Z=20*X**2+Y**2

x0=-5
y0=40

z=20*x**2+y**2


# dz/dx
dz_dx=40*x
dz_dy=2*y

# etha_x=0.05
# etha_y=0.05
#
# etha_x=0.015
# etha_y=0.015


# etha_x=0.015
# etha_y=0.05


num_iter=500

x=x0
y=y0
x_s=[]
y_s=[]
g_s=[]
etha=0.9

G_t=np.zeros((2,2))

def compute_grdient(x,y):
    dz_dx = 40 * x
    dz_dy = 2 * y
    g=np.array([dz_dx,dz_dy])
    g = g.reshape(g.shape[0], -1)
    return g


for i in np.arange(0,num_iter):
    # g_t=(dz/dx,dz/dy)
    dz_dx = 40 * x
    dz_dy = 2 * y
    g=np.array([dz_dx,dz_dy])
    g = g.reshape(g.shape[0], -1)
    g_s.append(g)

    G_tau=np.dot(g, np.transpose(g))
    G_t =G_t+G_tau

    diag_G_t_square= np.diag(np.power(G_t, 0.5))

    x = x-etha*g[0]/diag_G_t_square[0]
    y = y-etha*g[1]/diag_G_t_square[1]

    print("eta x:" ,etha*g[0]/diag_G_t_square[0])
    print("eta y:", etha * g[1] / diag_G_t_square[1])

    x_s.append(x)
    y_s.append(y)
    print(x)
    print(y)


levels=np.arange(np.min(Z), np.max(Z),10)
plt.contourf(X,Y,Z,levels=levels)
plt.plot(x_s,y_s, color='r')
plt.show()