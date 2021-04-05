import torch
import numpy as np

# https://neptune.ai/blog/pytorch-loss-functions
# https://github.com/behnamasadi/PythonTutorial/blob/master/Machine_Learning/nn/loss_functions.ipynb

input=torch.randn(3,4,requires_grad=True)
target=torch.randn(3,4)

print('######################## 1) Mean Absolute Error Loss ########################')


# sum of absolute differences between actual values and predicted values, loss(x,y)=|x-y|



mae_loss=torch.nn.L1Loss()
output=mae_loss(input,target)
output.backward()
print(output)


print('######################## 2) Mean Squared Error Loss ########################')

# loss(x,y)=(x-y)^2

mae_loss=torch.nn.MSELoss()
output=mae_loss(input,target)
output.backward()
print(output)


print('######################## 3) Negative Log-Likelihood Loss ########################')

# https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html

# size of input (N x C) is = 3 x 5
#N
number_of_item_in_training_set=3

#C
number_of_neuron_in_output_layer=5

input = torch.randn(number_of_item_in_training_set, number_of_neuron_in_output_layer, requires_grad=True)

# target should have only 3 elements (N=3) and every element in target should have 0 <= value < C=5
# because these are target lables. Target: (N)
target = torch.tensor([4, 3, 2])

m = torch.nn.LogSoftmax(dim=1)

# NLLLoss expects the inputs to be log probabilities
nll_loss = torch.nn.NLLLoss()
output = nll_loss(m(input), target)
output.backward()

print('input: \n', input)
print('target: \n', target)
print('output: \n', output)
print('m(input).exp()\n',m(input).exp())


print('######################## LogSoftmax example ########################')

m = torch.nn.LogSoftmax(dim=1)
input = torch.randn(2, 3)
output = m(input)

print('input: \n', input)
print('LogSoftmax(input): \n', output)
print('sum on output of softmax: \n', np.exp(output).sum(axis=1))


# Cross-Entropy Loss
# Hinge Embedding Loss
# Margin Ranking Loss
# Triplet Margin Loss
#Kullback-Leibler divergence

