# http://www.bnikolic.co.uk/blog/pytorch-detach.html
#detach() detaches the output from the computationnal graph. So no gradient will be backproped along this variable.
import torch
import torchviz

a=torch.tensor([1.2],requires_grad=True)
b=torch.tensor([2.2],requires_grad=True)
c=torch.tensor([0.2],requires_grad=True)
d=torch.tensor([0.8],requires_grad=True)
e=torch.tensor([7.],requires_grad=True)

f=d*((a+b)*(c))+e

f.backward()


print("a",a.grad)
print('b',b.grad)
print('c',c.grad)
print('d',d.grad)
print('e',e.grad)

graph=torchviz.make_dot(f)
graph.format='svg'
graph.save('images/graph')
graph.render()


#torch.no_grad says that no operation should build the graph.


a=torch.tensor([1.2],requires_grad=True)
b=torch.tensor([2.2],requires_grad=True)
c=torch.tensor([0.2],requires_grad=True)
d=torch.tensor([0.8],requires_grad=True)
e=torch.tensor([7.],requires_grad=True)

f=d*((a+b)*(c.detach()))+e
f.backward()


print("a",a.grad)
print('b',b.grad)
print('c',c.grad)
print('d',d.grad)
print('e',e.grad)


graph=torchviz.make_dot(f)
graph.format='svg'
graph.save('images/graph_detach')
graph.render()
