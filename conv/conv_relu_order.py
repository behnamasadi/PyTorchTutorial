# conv -> relu -> max pooling or conv -> max pooling -> relu?

# MaxPool(Relu(x)) = Relu(MaxPool(x))

# If you consider the final result, both orders [conv -> relu -> max pooling] and [conv -> max pooling -> relu]
# will have the same outputs. But if you compare the running time of 2 ways there will be a difference.

# Relu layer don’t change the size of the input. Let assume we use max pooling 2x2, so the size of input will
# be reduce by 2 in height and width when apply max poling layer ( [w, h, d] -> max_pooling_2x2 -> [w/2, h/2, d]).

# In case 1 we using relu -> max pooling the size of data will be:
#
# image[w, h, d] -> [[relu]] ->image[w, h, d]->[[max pooling]] -> image[w/2, h/2, d]
#
# In case 2 we using max pooling -> relu the size of data will be:
#
# image[w, h, d] ->[[max pooling]] -> image[w/2, h/2, d]-> [[relu]] -> image[w/2, h/2, d]
#
# image[w, h, d] -> [[relu]] vs image[w/2, h/2, d]-> [[relu]] : case 2 save 4 time computational cost than case 1
# in layer [[relu]] by using max pooling before relu.