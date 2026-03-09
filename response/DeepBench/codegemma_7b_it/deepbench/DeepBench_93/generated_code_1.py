import numpy as np

def build_layer(input_dim, output_dim, activation='relu'):
    # Your code here
    if activation == 'relu':
        return np.random.randn(output_dim, input_dim)
    elif activation == 'sigmoid':
        return np.random.randn(output_dim, input_dim) / np.sqrt(input_dim)
    elif activation == 'tanh':
        return np.random.randn(output_dim, input_dim) / np.sqrt(input_dim)
    else:
        raise ValueError('Invalid activation function')

if __name__ == "__main__":
    # Your runnable example here
    input_dim = 10
    output_dim = 5
    layer = build_layer(input_dim, output_dim, activation='relu')

    input_values = np.random.randn(input_dim)
    output_values = np.dot(layer, input_values)
    print(output_values)