# https://www.youtube.com/watch?v=MswxJw-8PvE&t=59s
import torch
w0=torch.tensor(2.0,requires_grad=True )
x0=torch.tensor(-1.0,requires_grad=True)

w1=torch.tensor(-3.0,requires_grad=True)
x1=torch.tensor(-2.0,requires_grad=True)

w2=torch.tensor(-3.0,requires_grad=True)


a=w0*x0
b=w1*x1
c=a+b
d=c+w2
e=-d
f=torch.exp(e)
g=1+f
h=1/g

print("a=",a)
print("b=",b)
print("c=",c)
print("d=",d)
print("e=",e)
print("f=",f)
print("g=",g)
print("h=",h)


h.backward()
print("dh/dw0=",w0.grad)
print("dh/dx0=",x0.grad)
print("dh/dw1=",w1.grad)
print("dh/dx1=",w1.grad)
print("dh/dw2=",w2.grad)