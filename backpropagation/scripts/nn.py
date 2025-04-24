import numpy as np
import mnist_loader
import matplotlib.pyplot as plt
class Network(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def evaluate(self, data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""

        # y is a number which is teh index of correct lables, for instance y=0 means the number in the image is 1
        # self.feedforward(x) return an array of size 10 in which the index of max value is the predeicted label
        # results is a list of tuples,
        results=[]
        for(x,y) in data:
            predicted_label=np.argmax(self.feedforward(x))
            actual_label=y
            results.append((predicted_label,actual_label))
        #test_results = [(np.argmax(self.feedforward(x)), y)  for (x, y) in data]
        #return sum(int(x == y) for (x, y) in test_results)

        return sum(int(predicted_label==actual_label) for (predicted_label,actual_label) in results)

    def sigmoid(self, z):
        return 1.0/(1+np.exp(-z))

    def sigmoid_prime(self, z):
        """Derivative of the sigmoid function."""
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def feedforward(self,a):
        for w,b in zip(self.weights,self.biases):
            a=self.sigmoid(np.dot(w,a)+b)
        return a

    def train(self, training_set, batch_size):
        self.training_set=training_set

        indices=np.arange(0,training_set.shape[0])
        np.random.shuffle(indices)
        print(indices)

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, a[0] contains a of the neurons in the first layer
        zs = [] # list to store all the z vectors, z[0] contains z of the neurons in the first layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)
        # backward pass
        #eq 1. 𝜹(𝐿)=(𝐚(𝐿)−𝐲)⊙𝜎′(𝐳(𝐿))
        delta_l = (activations[-1]- y) * self.sigmoid_prime(zs[-1])
        delta = (activations[-1] - y) * self.sigmoid_prime(zs[-1])
        #eq 2. ∂𝐜/∂𝐛(𝐿)=𝜹(𝐿)
        nabla_b[-1] = delta_l

        #eq. 3  𝜹(𝐿)⋅𝐚(𝐿−1)⊤
        nabla_w[-1] = np.dot(delta_l, activations[-2].transpose())

        # we have 3 layer that means 2 weight matrices, so the index of last weight matrix is 1 not 2
        # so we have to adjust the indices
        L=self.num_layers-2
        for l in np.arange(L,0,-1 ):
            #eq 4. 𝜹(𝐿−1)=∂𝑐/∂𝑧(𝐿−1)=((𝐖(𝐿))𝑇𝜹(𝐿))⊙𝜎′(𝑧(𝐿−1))

            delta_l_1=np.dot(np.transpose(self.weights[l]),delta_l)*self.sigmoid_prime(zs[l-1])

            # eq 5. ∂𝐜/∂𝐛(𝐿−1)=𝜹(𝐿−1)
            nabla_b[l-1]=delta_l_1

            #eq 6. ∂𝐜/∂𝐖(𝐿−1)=𝜹(𝐿−1)⋅𝐚(𝐿−2)⊤
            #again we have to adjust the indices so we use l-1 instead of activations[l-1]
            # because activations is a list of size 3 but nabla_w and nabla_b are list of size 2
            nabla_w[l-1]=np.dot(delta_l_1, np.transpose(activations[l-1]))

            delta_l=delta_l_1

        # for l in np.arange(2, self.num_layers):
        #     z = zs[-l]
        #     sp = self.sigmoid_prime(z)
        #     delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
        #     nabla_b[-l] = delta
        #     nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        #
        return (nabla_b, nabla_w)


    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs."""
        if test_data: n_test = len(test_data)
        n = len(training_data)

        epochs_number=[]
        result_accuracy=[]

        for j in np.arange(epochs):
            np.random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in np.arange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                data_evaluation=self.evaluate(test_data)
                print ("Epoch {0}: {1} / {2}".format(j, data_evaluation, n_test))
                epochs_number.append(j)
                result_accuracy.append(data_evaluation/n_test)



            else:
                print ("Epoch {0} complete".format(j))
        plt.plot(epochs_number, result_accuracy)
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plt.show()

    def update_mini_batch(self, mini_batch, eta):
        # nabla_b[0] contains the amount that should be added to the biases in the first layer
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:

            # delta_nabla_b[0] contains the gradient of the first layer from one single input,
            # we compute gradient from all inputs and make an average and add that average to the current value of
            # bias and weight

            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            #nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            #nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            for i in np.arange(len(nabla_b)):
                nabla_b[i]=nabla_b[i]+delta_nabla_b[i]
                nabla_w[i]=nabla_w[i]+delta_nabla_w[i]

        for i in np.arange(len(self.weights)):
            self.weights[i] = self.weights[i]- eta *nabla_w[i] /len(mini_batch)
            self.biases[i] = self.biases[i] - eta * nabla_b[i] / len(mini_batch)

        #self.weights = [w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
        #self.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]

sizes=[2,4,3,5]
# The index of -1 refers to the last item, -2 to the second last item and so on.
#  p   r   o   b   e
#  0   1   2   3   4
# -5  -4  -3  -2  -1

# weights[0] is a Numpy matrix storing the weights connecting the first and second layers of neurons.
# weights[0]=w is a matrix such that wjk is the weight for the connection between the kth neuron in the first layer,
# and the jth neuron in the second layer.
# np.random.randn generates Gaussian distributions with mean 0 and standard deviation 1.

# nn = Network(sizes)
# print(nn.weights)
# print(nn.biases)
# for b,w in zip(nn.biases,nn.weights):
#     print("w:", w.shape)
#     print("b: ", b.shape)

#input = np.random.randn(sizes[0],1)
#print(input.shape)
#print(input)
#print(nn.feedforward(input))

#print("xxxx", np.random.randn(3,1)*np.random.randn(3,1))

# batch_size=10
# input_layer_size=sizes[0]
# output_layer_size=sizes[-1]
# number_of_trianing_samples=100

#training_data=np.random.randn(input_layer_size,1)
#labels=np.random.randn(output_layer_size,1)
#nn.train(training_set,batch_size)

training_data , validation_data , test_data = mnist_loader.load_data_wrapper()

# we have tree layers, the input layer has 784 neurons, the hidden layer has 30 neurons, the output has 10 neurons
# so we have two weight matrices, two biases and two z vector
net = Network([784, 30, 10])



epochs=30
mini_batch_size=10
# eta is learning rate for gradient descent
eta=3.0
net.SGD( training_data, epochs, mini_batch_size, eta, test_data)





from PIL import Image
pixels=[
    [0, 1, 0, 1, 0, 1, 0, 1, ],
    [1, 0, 1, 0, 1, 0, 1, 0, ],
    [0, 1, 0, 1, 0, 1, 0, 1, ],
    [1, 0, 1, 0, 1, 0, 1, 0, ],
    [0, 1, 0, 1, 0, 1, 0, 1, ],
    [1, 0, 1, 0, 1, 0, 1, 0, ],
    [0, 1, 0, 1, 0, 1, 0, 1, ],
    [1, 0, 1, 0, 1, 0, 1, 0, ],
]

g = np.asarray(dtype=np.dtype('uint8'), a=pixels)
g=np.round(255*g)
image = Image.fromarray(g.astype(np.uint8), mode='L')
image.save('checker.png')

image=Image.open('checker.png')
print(np.array(image))

z=1.5
def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))

def sigmoid_prime( z):
    """Derivative of the sigmoid function."""
    return sigmoid(z) * (1 - sigmoid(z))

print(sigmoid(z))
print(sigmoid_prime(4))

