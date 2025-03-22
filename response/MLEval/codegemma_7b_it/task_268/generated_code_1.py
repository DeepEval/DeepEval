import numpy as np

class NeuralNetwork:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.inputSize = X.shape[1]
        self.hiddenSize = 6
        self.outputSize = y.shape[1]

        # Initialize weights randomly
        self.wh = np.random.random((self.inputSize, self.hiddenSize))
        self.bh = np.random.random((1, self.hiddenSize))
        self.wout = np.random.random((self.hiddenSize, self.outputSize))
        self.bout = np.random.random((1, self.outputSize))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def dsigmoid(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def tanh(self, x):
        return np.tanh(x)

    def dtanh(self, x):
        return 1 - np.tanh(x) ** 2

    def forward(self):
        self.z = np.dot(self.X, self.wh) + self.bh
        self.hidden = self.tanh(self.z)
        self.z2 = np.dot(self.hidden, self.wout) + self.bout
        self.output = self.sigmoid(self.z2)

    def backprop(self):
        # Calculate error gradients for output layer
        error = self.y - self.output
        d_output = error * self.dsigmoid(self.output)

        # Calculate error gradients for hidden layer
        error_hidden = np.dot(d_output, self.wout.T)
        d_hidden = error_hidden * self.dtanh(self.hidden)

        # Update weights and biases
        self.wout += np.dot(self.hidden.T, d_output)
        self.bout += np.sum(d_output, axis=0, keepdims=True)
        self.wh += np.dot(self.X.T, d_hidden)
        self.bh += np.sum(d_hidden, axis=0, keepdims=True)

    def train(self, iterations):
        for i in range(iterations):
            self.forward()
            self.backprop()

# Example usage:

# Create a neural network
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])
nn = NeuralNetwork(X, y)

# Train the neural network
nn.train(1000)

# Get the output
output = nn.output

# Print the output
print(output)