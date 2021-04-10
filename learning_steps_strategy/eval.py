# https://stackoverflow.com/questions/60018578/what-does-model-eval-do-in-pytorch
# model.eval() is a kind of switch for some specific layers/parts of the model
# (i.e. Dropouts Layers, BatchNorm Layers etc) that behave  differently during training and inference (evaluating) time.
#
# You need to turn off them during model evaluation, and .eval() will do it for you.
# In addition, the common practice for evaluating/validation is using torch.no_grad() in pair with model.eval() to turn off gradients computation:


# # evaluate model:
# model.eval()
#
# with torch.no_grad():
#     ...
#     out_data = model(data)


# training step

# model.train()
