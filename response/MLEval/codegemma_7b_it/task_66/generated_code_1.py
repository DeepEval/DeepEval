import numpy as np

def method():
    # Define input and output data
    input_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    output_data = np.array([[0], [1], [1], [0]])

    # Define hyperparameters
    hidden_layer_neurons = 10
    input_neurons = 2
    output_neurons = 1
    learning_rate = 0.5

    # Initialize weights randomly
    hidden_weights = np.random.rand(input_neurons, hidden_layer_neurons)
    output_weights = np.random.rand(hidden_layer_neurons, output_neurons)

    # Define activation function
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Define derivative of activation function
    def sigmoid_derivative(x):
        return sigmoid(x) * (1 - sigmoid(x))

    # Training loop
    for epoch in range(10000):
        # Forward propagation through hidden layer
        hidden_layer_input = np.dot(input_data, hidden_weights)
        hidden_layer_output = sigmoid(hidden_layer_input)

        # Forward propagation through output layer
        output_layer_input = np.dot(hidden_layer_output, output_weights)
        output = sigmoid(output_layer_input)

        # Calculate error
        error = output_data - output

        # Backpropagation through output layer
        output_layer_error = error * sigmoid_derivative(output)

        # Backpropagation through hidden layer
        hidden_layer_error = np.dot(output_layer_error, output_weights.T) * sigmoid_derivative(hidden_layer_output)

        # Update weights
        output_weights += learning_rate * np.dot(hidden_layer_output.T, output_layer_error)
        hidden_weights += learning_rate * np.dot(input_data.T, hidden_layer_error)

    # Print final output
    print(output)

# Call the method
method()