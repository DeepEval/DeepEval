import numpy as np

def method(n_inputs, n_hidden):
    # Initialize weights and biases for the hidden layer
    W1 = np.random.rand(n_inputs, n_hidden)
    b1 = np.zeros((n_hidden, 1))

    # Initialize weights and biases for the output layer
    W2 = np.random.rand(n_hidden, 1)
    b2 = np.zeros((1, 1))

    # Define the sigmoid activation function
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Define the derivative of the sigmoid activation function
    def sigmoid_derivative(x):
        return x * (1 - x)

    # Forward pass
    hidden_layer = sigmoid(np.dot(X, W1) + b1)
    output = sigmoid(np.dot(hidden_layer, W2) + b2)

    # Backward pass
    d_output = np.array([[0.5]])
    d_hidden = np.dot(d_output, W2.T) * sigmoid_derivative(hidden_layer)
    d_b1 = np.sum(d_hidden, axis=0, keepdims=True)
    d_b2 = np.sum(d_output, axis=0, keepdims=True)
    d_W1 = np.dot(X.T, d_hidden)
    d_W2 = np.dot(hidden_layer.T, d_output)

    # Print the results
    print("Weights and biases for the hidden layer:")
    print(W1)
    print(b1)
    print("Weights and biases for the output layer:")
    print(W2)
    print(b2)
    print("Derivatives of the weights and biases for the hidden layer:")
    print(d_W1)
    print(d_b1)
    print("Derivatives of the weights and biases for the output layer:")
    print(d_W2)
    print(d_b2)

    # Return the final output
    return output

# Call the method for validation
output = method(2, 3)