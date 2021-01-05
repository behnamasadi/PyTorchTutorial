import torch
from torchviz import make_dot, make_dot_from_trace


if torch.cuda.is_available():
    dev=torch.device("cuda")
else:
    dev = torch.device("cpu")

N,D_in,H, D_out=64, 1000, 100, 10
learning_rate=1e-6

x=torch.randn(N,D_in,device=dev)
y=torch.randn(N,D_out,device=dev)

model1=torch.nn.Sequential(torch.nn.Linear(D_in,H),
                    torch.nn.ReLU(),
                    torch.nn.Linear(H,D_out)).to(dev)

model2=torch.nn.Sequential()
model2.add_module('w0', torch.nn.Linear(D_in,H))
model2.add_module('ReLU', torch.nn.ReLU())
model2.add_module('w1', torch.nn.Linear(H,D_out))



loss_function=torch.nn.MSELoss(reduction='sum')

number_of_iterations=500

y_predict = model1(x)
#print(torch.sum(model2(x) - y_predict))
print(model2(x)[0] - y_predict[0])

make_dot(model1(x), params=dict(model1.named_parameters())).render("model1", format="svg")
make_dot(model2(x), params=dict(model2.named_parameters())).render("model2", format="svg")



for i in range(number_of_iterations):
    y_predict=model1(x)
    loss = loss_function(y_predict, y)
    print(i, loss.item())
    loss.backward()
    with torch.no_grad():
        for param in model1.parameters():
            param.data -= learning_rate * param.grad

    model1.zero_grad()


