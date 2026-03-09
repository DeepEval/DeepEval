import numpy as np

def build_layer(input_size, hidden_size, num_layers, activation_function=None):
    # Initialize the layer list
    layer_list = []

    # Add the input layer
    layer_list.append(input_size)

    # Add the hidden layers
    for i in range(num_layers):
        layer_list.append(hidden_size)

    # Add the output layer
    layer_list.append(hidden_size)

    # Return the layer list
    return layer_list

# Example usage
if __name__ == "__main__":
    # Define the input size, hidden size, and number of layers
    input_size = 28 * 28
    hidden_size = 10
    num_layers = 3

    # Build the layer list
    layer_list = build_layer(input_size, hidden_size, num_layers)

    # Print the layer list
    print(layer_list)

    # Example input values
    x = np.random.rand(input_size)

    # Compute the output values
    output = np.zeros(hidden_size)
    for i in range(num_layers):
        output = np.dot(output, layer_list[i])
        output = activation_function(output)

    # Print the output values
    print(output)