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


num_iter=200

x=x0
y=y0
x_s=[]
y_s=[]

etha=0.15

G_t=np.zeros((2,2))
gamma=0.99

v_x=0
v_y=0

for i in np.arange(0,num_iter):
    # g_t=(dz/dx,dz/dy)
    dz_dx = 40 * x
    dz_dy = 2 * y

    dz_dx_square=dz_dx**2
    dz_dy_square = dz_dy ** 2

    v_x = gamma*v_x + (1 - gamma)*dz_dx_square
    v_y = gamma*v_y + (1 - gamma)*dz_dy_square

    x = x-etha*dz_dx/np.sqrt(v_x)
    y = y-etha*dz_dy/np.sqrt(v_y)

    x_s.append(x)
    y_s.append(y)
    print(x)
    print(y)


levels=np.arange(np.min(Z), np.max(Z),10)
plt.contourf(X,Y,Z,levels=levels)
plt.plot(x_s,y_s, color='r')
plt.show()