#https://www.youtube.com/watch?v=28BMpgxn_Ec
#https://www.youtube.com/watch?v=42zJ5xrdOqo
#https://www.youtube.com/watch?v=b2eULzgZuo8
#It is a second order method in comparision with gradient descent which is a first order


import numpy as np
import matplotlib.pyplot as plt

# f(x)=x^3-x^2+1
#f(x+Δx)≈f(x) + f'(x)*Δx +1/2 f''(x)*Δx^2
#f(x+Δx)=0  =>  f(x)' +f''(x)*Δx =0  => Δx=-f(x)'/f''(x)
#x_n+1=x_n+Δx =xn-f(x)'/f''(x)


def F(x):
    return x**3-x**2+1

def F_1st_derivative(x):
    return 3*x**2-2*x

def F_2nd_derivative(x):
    return 6*x-2


#Higher dimensions
# x_n+1=x_n  -γ * [H*f(x_n)]^-1 ▼f(x_n)


# γ: step_size
# H: Hessian matrix
# ▼f(x_n):  gradient of f
start=-10 
end=10
step=0.2
X=np.arange(start,end,step) 


x_initial=5
x=x_initial
number_of_iterations=100

list_of_x=[]
lamba=0.1
#x_n+1=x_n+Δx =xn-f(x)'/f''(x)
for i in range(number_of_iterations):
    list_of_x.append(x)
    #print("x:",x)
    #print("diff:",lamba*F_1st_derivative(x)/F_2nd_derivative(x))
    x=x-lamba*F_1st_derivative(x)/F_2nd_derivative(x)

print(x)
    
plt.plot(X,F(X))
plt.scatter(list_of_x,F( np.asarray(list_of_x)  ), color='red')
plt.show()


#Wolfe conditions ?

