import numpy as np
import matplotlib.pyplot as plt


"""size of layer, i.e. [2,3,3,2] means

        0   0
    0   0   0   0
    0   0   0   0

    so weights would be: [3,2] , [3,3] , [2,3] 
    and biases would be  [3] , [3] , [2]  


"""


class NeuralNetwork():

    def __init__(self, sizes):
        rnd = np.random.default_rng(seed=42)

        self.sizes = sizes
        self.weight_dims = zip(sizes[1:], sizes[:-1])
        self.weights = []
        self.delta_Ls = [None] * (len(sizes) - 1)  # no delta for input layer

        for weight_dim in self.weight_dims:
            # print("weight dimension", weight_dim)
            weight = rnd.random(weight_dim)
            # print(weight)
            self.weights.append(weight)

        self.bias_dims = sizes[1:]
        self.biases = []
        for bias_dim in self.bias_dims:
            # print("bias dimension:", bias_dim, ",1")
            bias = rnd.random([bias_dim, 1])
            # print(bias)
            self.biases.append(bias)

    def sigmoidFunction(self, x):
        return 1 / (1 + np.exp(-x))

    # def feedforward(self, x):
    #     a = x
    #     for w, b in zip(self.weights, self.biases):
    #         z = np.dot(w, a) + b
    #         a = self.activationFunction(z)
    #     return a

    def feedforward(self, X):
        """
        X: input matrix of shape (input_size, batch_size)
        Returns: activation of final layer, shape (output_size, batch_size)
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)  # Handle single input

        # self.zs 3 x batch_size, 3 x batch_size, 2 x batch_size
        # self.activations  2 x batch_size 3 x batch_size, 3 x batch_size, 2 x batch_size

        self.zs = []  # pre-activations
        # list of activations, starting with input layer
        self.activations = [X]

        a = X

        for w, b in zip(self.weights, self.biases):
            # b: shape (n, 1), needs to be broadcasted across batch dimension
            z = np.matmul(w, a) + b
            print("z.shape:", z.shape)
            self.zs.append(z)
            a = self.sigmoidFunction(z)
            print("a.shape:", a.shape)
            self.activations.append(a)

        return a

    def lossFunction(self, Y_true, Y_pred):
        # or cross-entropy if you're doing classification.
        return np.mean((Y_pred - Y_true) ** 2)

    def activationFunction(self, x):
        return self.sigmoidFunction(x)

    def backpropagation(self,):
        # delta_output_layer=
        pass

    def sigmoidDerivative(self, x):
        s = self.sigmoidFunction(x)
        return s * (1 - s)

    def outputLayerError(self, y_true):
        """
        y_true: true labels, shape (output_size, batch_size)
        Uses the stored activations and z values from last feedforward pass
        """

        print(len(self.activations))

        print(self.activations[0].shape)

        print(len(self.zs))

        aL = self.activations[-1]
        zL = self.zs[-1]
        delta_L = (aL - y_true) * self.sigmoidDerivative(zL)

        print("delta_L.shape:", delta_L.shape)

        self.delta_Ls[-1] = delta_L

        return delta_L

    def hiddenLayerError(self):
        # Backpropagate the error from output to first hidden layer
        for L in reversed(range(len(self.delta_Ls) - 1)):  # from L-2 to 0
            zL = self.zs[L]
            delta_next = self.delta_Ls[L + 1]
            w_next = self.weights[L + 1]
            delta_L = (w_next.T @ delta_next) * self.sigmoidDerivative(zL)
            self.delta_Ls[L] = delta_L

    def computeGradients(self):
        """
        Returns:
            grad_W: list of gradients for weights
            grad_b: list of gradients for biases
        """
        batch_size = self.activations[0].shape[1]
        grad_W = []
        grad_b = []

        for l in range(len(self.weights)):
            delta = self.delta_Ls[l]
            a_prev = self.activations[l]

            dW = delta @ a_prev.T / batch_size  # shape: (n_l, n_{l-1})
            db = np.mean(delta, axis=1, keepdims=True)  # shape: (n_l, 1)

            grad_W.append(dW)
            grad_b.append(db)

        return grad_W, grad_b

    def updateParameters(self, grad_W, grad_b, learning_rate):
        for l in range(len(self.weights)):
            self.weights[l] -= learning_rate * grad_W[l]
            self.biases[l] -= learning_rate * grad_b[l]


sizes = [2, 3, 3, 2]
nn = NeuralNetwork(sizes)

x = np.array([1, 4]).reshape(-1, 1)
y = np.array([1, 4]).reshape(-1, 1)


# print("------------")
# y_pred = nn.feedforward(x)
# print(y_pred)


# Batch of 3 samples
X = np.array([
    [1, 2, 3, 6, 2],
    [4, 5, 6, 4, 1]
])  # Shape: (2, 3)

Y = np.array([
    [0, 1, 1, 1, 0],
    [1, 0, 0, 1, 1]
])  # Shape: (2, 3)

y_pred = nn.feedforward(X)
nn.lossFunction(Y, y_pred)
print(y_pred)

nn.outputLayerError(Y)

for w in nn.weights:
    print(w.shape)

output = nn.feedforward(X)
nn.outputLayerError(Y)
nn.hiddenLayerError()

for i, d in enumerate(nn.delta_Ls):
    print(f"delta layer {i+1}: shape={d.shape}")


grad_W, grad_b = nn.computeGradients()
nn.updateParameters(grad_W, grad_b, learning_rate=0.1)

# Check updated weights
print("Updated weights:")
for w in nn.weights:
    print(w)


# 3

# XOR dataset
X = np.array([
    [0, 0, 1, 1],
    [0, 1, 0, 1]
])  # shape: (2, 4)

Y = np.array([
    [0, 1, 1, 0]
])  # shape: (1, 4)

# Create network
nn = NeuralNetwork([2, 3, 1])

# Training parameters
epochs = 10000
learning_rate = 1.0
losses = []

for epoch in range(epochs):
    # Forward pass
    y_pred = nn.feedforward(X)

    # Compute loss
    loss = nn.lossFunction(Y, y_pred)
    losses.append(loss)

    # Backward pass
    nn.outputLayerError(Y)
    nn.hiddenLayerError()

    # Gradients and update
    grad_W, grad_b = nn.computeGradients()
    nn.updateParameters(grad_W, grad_b, learning_rate)

    # Print every 1000 epochs
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.6f}")

# Final prediction
print("\nFinal predictions:")
print(nn.feedforward(X))

# Plot loss
plt.plot(losses)
plt.title("Loss over epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()


print(nn.feedforward(X))
