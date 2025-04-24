import numpy as np
import matplotlib.pyplot as plt



#rate=(V_max*S)/(K_M+ S)
def F(S, V_max,K_M):
    
    rate=(V_max*S)/(K_M+ S)
    return rate


#b1=V_max,  b2=K_M


def Jacobian_r(S,b):

    #print( np.shape( -S/(b[1,0]+S)    )   )
    #print( np.shape(  b[0,0]*S/(b[1,0]+S)**2)  )  
    #print( np.vstack( ( -S/(b[1,0]+S)   ,  b[0,0]*S/(b[1,0]+S)**2) )  )  
    #return np.array(  [  [-S/(b[1,0]+S)] ,  [b[0,0]*S/(b[1,0]+S)**2]   ])
    return np.hstack( ( -S/(b[1,0]+S)   ,  b[0,0]*S/(b[1,0]+S)**2) )

#Residual =y-F(S, V_max,K_M)
def Residual(S,b,Y):
    r=Y-F(S,b[0,0],b[1,0])
    return r 


S=np.array(  [ [0.038]  , [0.194] ,  [0.425]  ,   [0.626] ,   [1.253] ,   [2.500] ,   [3.740] ])
Rate=np.array(  [ [0.050]  , [0.127] ,  [ 0.094]  ,   [0.2122 ] ,   [0.2729 ] ,   [ 0.2665] ,   [0.3317 ] ])





#Initial Guess:
b=np.array( [[0.9],[0.2]])
number_of_iteration=10

for i in range(number_of_iteration):
    J_r=Jacobian_r(S,b)
    J_r_T=np.transpose(J_r)
    
    #print(J_r)
    
    #print(J_r.shape)
    #print(J_r_T.shape)
    
    #print(Residual(S,b,Rate))
    b-= np.linalg.inv(J_r_T @J_r) @J_r_T @ Residual(S,b,Rate) 
    
                    
print(b)

start=0.0
end=6
step=.01

V_max=b[0,0]
K_M=b[1,0]

x=np.arange(start,end,step)
y=F(x, V_max,K_M)
#print(x.shape)
#print(y.shape)

plt.plot(x,y)
plt.scatter(S,Rate,color='red')
plt.show()
