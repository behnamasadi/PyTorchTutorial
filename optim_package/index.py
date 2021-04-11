import torch

if torch.cuda.is_available():
    dev=torch.device("cuda")
else:
    dev = torch.device("cpu")

learning_rate=1e-6
N,D_in,H,D_out=64,1000,100,10
x=torch.randn(N,D_in)
y=torch.randn(H,D_out)

model=torch.nn.Sequential()
model.add_module('w0',torch.nn.Linear(D_in,H))
model.add_module('relu',torch.nn.ReLU())
model.add_module('w1',torch.nn.Linear(H,D_out))


print(x.shape)
print(y.shape)


for param in model.parameters():
    print(type(param), param.size())



loss_function=torch.nn.MSELoss(reduction='sum')

optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate)

number_of_iterations=500
for i in range(number_of_iterations):
    y_predict=model(x)
    loss=loss_function(y_predict,y)
    print(i, loss.item())
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()


