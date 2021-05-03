# lecture 6 cs231n 2017 1:20:19


# 2) Choose architecture
# 2-1) Weight initialization
# I) xavier
# II) kaiming-he
# 2-2) Check if the loss is reasonable
# I) Disable regularization, do a forward and check the loss
# II) Crank up the regularization and the loss should goes up,
# III) Feed a small portion of data, turn off regularization, it should easily overfit
# and loss goes to zero, train accuracy =1

# batch normalization
# drop out

# https://nextjournal.com/gkoehler/pytorch-mnist
# https://ernie55ernie.github.io/machine%20learning/2018/03/28/weight-initialization-for-neural-network.html
# https://towardsdatascience.com/useful-plots-to-diagnose-your-neural-network-521907fa2f45
# https://cs231n.github.io/neural-networks-3/

# plotting
####################### The Data #######################
#1)  After training a classification model, there might be situations where the output of the model completely
# or mostly belongs to one class ie, the situation where the model is biased. This is primarily due to an imbalanced dataset.
# 2) not having enough data to support the problem statement.


####################### Loss Curve #######################
# https://cs231n.github.io/neural-networks-3/
# It is said that it is ideal to plot loss across epochs rather than iteration.

####################### Accuracy Curve #######################


####################### Uncertainty #######################


# don’t use autograd of not necessary (use with torch.no_grad() if possible)
# only push tensors to GPU, if they are actually needed
# try to avoid loops over tensor dimensions (slows things down)
# try to free graphs as soon as possible (use detach or item whenever you can) to avoid memory leaks
