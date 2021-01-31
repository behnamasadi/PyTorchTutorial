# During training, randomly zeroes some of the elements of the input tensor with probability p using samples
# from a Bernoulli distribution. Each channel will be zeroed out independently on every forward call.


# This has proven to be an effective technique for regularization and preventing the co-adaptation of neurons
# co-adaptation refers to when different hidden units in a neural networks have highly correlated behavior.
# It is better for computational efficiency and the the model’s ability to learn a general representation
# if hidden units can detect features independently of each other.

# Improving neural networks by preventing  co-adaptation of feature detectors
import torch
p=0.2
m = torch.nn.Dropout(p=p)
input=torch.randn(2,3)
output=m(input)
print(input)
print(output)
# the outputs are scaled by a factor of 1/(1-p) during training.
print(output*(1-p))
