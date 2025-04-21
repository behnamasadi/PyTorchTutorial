import numpy as np
import matplotlib.pyplot as plt

number_of_training_data=1000
D=np.random.randn(number_of_training_data,500)
hidden_layer_sizes=[500]*10
activation_functions=['relu']*len(hidden_layer_sizes)
print(hidden_layer_sizes)
acts={'relu': lambda x:  np.maximum(0,x), 'tanh':lambda x:np.tanh(x)}
Zs={}
#coefficient=0.01
coefficient=1.0
for i in np.arange(len(hidden_layer_sizes)):
    if i==0:
        X=D
    else:
        X=Zs[i-1]
    fan_in=X.shape[1]
    fan_out=hidden_layer_sizes[i]
    W=np.random.randn(fan_in,fan_out)*coefficient/np.sqrt(fan_in/2)
    #W = np.random.randn(fan_in, fan_out) * coefficient
    #print(W.shape)
    #H=np.matmul(X,W)
    H=np.dot(X,W)
    H=acts[activation_functions[i]](H)
    Zs[i]=H
print("input mean and cov",np.mean(D), np.std(D))
print(Zs[3].shape)
# output for every item in the tarining set is in a row so we have to compute
mat=np.random.randint(size=(2,3),low=1, high=10)
print(np.mean(mat))
print(np.mean(mat,axis=0).shape)
print(np.mean(mat,axis=1).shape)

layer_means=[np.mean(z) for _,z in Zs.items()]
layer_stds=[np.std(z) for _,z in Zs.items()]

plt.figure()
plt.subplot(121)
plt.plot(list(Zs.keys()),layer_means,'ob-')
plt.title('layer mean')
plt.subplot(122)
plt.plot(list(Zs.keys()),layer_stds,'or-')
plt.title('layer std')


#plt.figure(figsize=(20,10))
plt.figure()
for i,Z in Zs.items():
    plt.subplot(1,len(Zs),i+1)
    plt.hist(Z.ravel(),30,range=(-1,1))
plt.show()
