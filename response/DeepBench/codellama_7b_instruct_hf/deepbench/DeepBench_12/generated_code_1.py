import numpy as np
import torch
import torch.nn.functional as F

def get_recommended_gain(nonlinearity, param=None):
    """
    Return the recommended gain value for the given nonlinearity function.
    The values are as follows:
        Linear/Identity: 1
        Conv{1,2,3}D: 1
        Sigmoid: 1
        Tanh: 1
        ReLU: sqrt(2)
        Leaky ReLU: sqrt(2/1 + negative_slope^2)
        SELU: 3/4
    """
    if nonlinearity == 'linear':
        return 1
    elif nonlinearity == 'conv1d':
        return 1
    elif nonlinearity == 'conv2d':
        return 1
    elif nonlinearity == 'conv3d':
        return 1
    elif nonlinearity == 'sigmoid':
        return 1
    elif nonlinearity == 'tanh':
        return 1
    elif nonlinearity == 'relu':
        return np.sqrt(2)
    elif nonlinearity == 'leaky_relu':
        return np.sqrt(2 / (1 + param**2))
    elif nonlinearity == 'selu':
        return 3/4
    else:
        raise ValueError(f"Invalid nonlinearity: {nonlinearity}")

# Example usage
if __name__ == "__main__":
    # Create sample input values
    input_values = torch.randn(10, 10)

    # Call the function
    gain = get_recommended_gain('relu')

    # Print the results
    print(gain)