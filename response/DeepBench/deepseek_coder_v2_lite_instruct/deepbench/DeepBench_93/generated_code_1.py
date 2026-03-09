import numpy as np

def build_layer(input_size, output_size, activation=None):
    # Initialize weights and biases
    weights = np.random.randn(input_size, output_size) * 0.01
    biases = np.zeros((1, output_size))

    # Define the layer
    def layer(input_data):
        # Calculate the linear combination
        linear_output = np.dot(input_data, weights) + biases
        # Apply activation function if specified
        if activation is None:
            return linear_output
        else:
            return activation(linear_output)
    
    return layer

if __name__ == "__main__":
    # Sample input values
    input_size = 4
    output_size = 3
    activation = np.tanh  # Example activation function

    # Create a sample input
    input_data = np.random.randn(1, input_size)

    # Build the layer
    layer = build_layer(input_size, output_size, activation)

    # Call the layer function
    output = layer(input_data)

    # Print the results
    print("Input:\n", input_data)
    print("Output:\n", output)