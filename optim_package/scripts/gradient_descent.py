# Behnam Asadi 
# http://ros-developer.com
# To see the function that we are working on visit:
# http://ros-developer.com/2017/05/07/gradient-descent-method-for-finding-the-minimum/
# or simply put the following latex code in a latex doc:
# $$ z= -( 4 \times e^{- ( (x-4)^2 +(y-4)^2 ) }+ 2 \times e^{- ( (x-2)^2 +(y-2)^2 ) } )$$


import matplotlib.pyplot as plt
import numpy as np



def objective_function(x,y):
    z=-( 4*np.exp(-(x-4)**2 - (y-4)**2)+2*np.exp(-(x-2)**2 - (y-2)**2) )
    return z


def f_prim(x,y):
    f_x=-( (-2)*(x-4)*4*np.exp(-(x-4)**2 - (y-4)**2)    +   (-2)*(x-2)*2*np.exp(-(x-2)**2 - (y-2)**2) )
    f_y=-( (-2)*(y-4)*4*np.exp(-(x-4)**2 - (y-4)**2)    +   (-2)*(y-2)*2*np.exp(-(x-2)**2 - (y-2)**2) )
    return [f_x,f_y]



# The starts point for the algorithm:
X_old=4
Y_old=5

X_new=X_old
Y_new=Y_old


# learning rate
etha=0.1

# stop criteria
precision = 0.01

x_path_to_max=[]
y_path_to_max=[]
z_path_to_max=[]


while True:
    ret_val = f_prim(X_new, Y_new)
    X_new = X_old - etha * ret_val[0]
    Y_new = Y_old - etha * ret_val[1]
    print(ret_val)

    #print("X_old, Y_old", X_old,Y_old )
    #print("X_new, Y_new", X_new,Y_new )

    x_path_to_max.append(X_new)
    y_path_to_max.append(Y_new)
    z = objective_function(X_new, Y_new)
    z_path_to_max.append(z)

    distance=np.sqrt( (X_new-X_old)**2 + (Y_new-Y_old)**2 )
    X_old = X_new
    Y_old = Y_new
    print("distance: ",distance)
    if(distance < precision):
        break



print(x_path_to_max)
print(y_path_to_max)
print(z_path_to_max)
    
    

x = np.linspace(-2,10,200)
y = np.linspace(-2,10,200)

X, Y = np.meshgrid(x,y)

Z=objective_function(X,Y)


#Make a 3D plot
fig = plt.figure()
ax = fig.gca(projection='3d')

ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
#line1=ax.plot(x_path_to_max,y_path_to_max,z_path_to_max)
line1=ax.scatter(x_path_to_max,y_path_to_max,z_path_to_max,marker='o',c='r')

# linestyle:  '-', '--', '-.', ':', 'None', ' ', '', 'solid', 'dashed', 'dashdot', 'dotted'
plt.setp(line1,color='g',linestyle=':',linewidth=4.5)


ax.plot_surface(X, Y, Z,linewidth=0,cmap='coolwarm')



plt.show()


