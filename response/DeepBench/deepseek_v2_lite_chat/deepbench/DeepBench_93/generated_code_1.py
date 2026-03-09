import numpy as np

def build_layer(input_size, num_nodes):
    # Initialize weights and biases for each node
    weights = np.random.randn(input_size, num_nodes)
    biases = np.random.randn(num_nodes)
    
    # Function to apply the layer
    def layer(inputs):
        # Compute the weighted sum of the inputs and biases
        weighted_sum = np.dot(inputs, weights) + biases
        # Apply activation function (e.g., ReLU)
        outputs = np.maximum(weighted_sum, 0)
        return outputs
    
    return layer

if __name__ == "__main__":
    # Create a sample input tensor
    input_tensor = np.array([1, 2, 3, 4, 5])
    
    # Number of nodes in the hidden layer
    num_hidden_nodes = 3
    
    # Build the layer
    layer = build_layer(len(input_tensor), num_hidden_nodes)
    
    # Apply the layer to the input tensor
    output_tensor = layer(input_tensor)
    
    # Print the output tensor
    print("Output Tensor:", output_tensor)