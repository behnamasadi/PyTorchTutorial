# https://spin.atomicobject.com/2014/06/24/gradient-descent-linear-regression/

from numpy import genfromtxt, meshgrid
import csv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


def error_for_give_m_b(m, b, X, Y):
    total_error = 0
    for i in range(len(X)):
        total_error = total_error + (Y[i] - (m*X[i]+b))**2

    return total_error/float(len(X))


def stepGradient(b_current, m_current, X, Y, learningRate):
    b_gradient = 0
    m_gradient = 0
    N = float(len(X))
    for i in range(0, len(X)):
        b_gradient = b_gradient - (2/N)*(Y[i] - (m_current*X[i] + b_current))
        m_gradient = m_gradient + - \
            (2/N)*X[i]*(Y[i] - (m_current*X[i]+b_current))
        # print 'm_gradient: ', m_gradient

    new_b = b_current - (learningRate*b_gradient)
    new_m = m_current - (learningRate*m_gradient)
    return [new_b, new_m]


with open('data.csv', 'rU') as csvfile:
    readit = csv.reader(csvfile, delimiter=',')
    csvfile_data = list(csvfile)


my_data = genfromtxt('data.csv', delimiter=',')


X = my_data[:, 0]
Y = my_data[:, 1]
# X=X+50
# Y=Y+50


error = []
b_current = 0
m_current = 1
learningRate = 0.0001
precision = 0.000001
new_b, new_m = stepGradient(b_current, m_current, X, Y, learningRate)
while min(abs(new_b-b_current), abs(new_m-m_current)) > precision:
    # print 'abs(new_b-b_current): ', abs(new_b-b_current)
    # print 'abs(new_m-m_current): ', abs(new_m-m_current)

    error.append(error_for_give_m_b(new_m, new_b, X, Y))

    b_current = new_b
    m_current = new_m
    new_b, new_m = stepGradient(new_b, new_m, X, Y, learningRate)

print(new_b)
print(new_m)


x0 = -10
y0 = new_m*x0+new_b

x1 = 100
y1 = new_m*x1+new_b


plt.scatter(X, Y)
plt.plot([x0, x1], [y0, y1], color='red')
plt.show()


plt.plot(error)
plt.show()


m = np.arange(-5, 5, 0.5)
b = np.arange(-5, 10, 0.5)
M, B = np.meshgrid(m, b)

error = error_for_give_m_b(M, B, X, Y)

fig = plt.figure()
ax = fig.gca(projection='3d')

# plot_surface
surf = ax.plot_surface(M, B, error, rstride=1, cstride=1,
                       cmap=cm.hot, linewidth=0, antialiased=False)
# ax.set_zlim(-1.01, 10000.01)

# ax.zaxis.set_major_locator(LinearLocator(10))
# ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
