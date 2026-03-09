import torch.nn.functional as F

def get_nonlinearity_gain(nonlinearity, param=None):
    if nonlinearity == 'linear' or nonlinearity == 'identity':
        return 1
    elif nonlinearity == 'conv1d' or nonlinearity == 'conv2d' or nonlinearity == 'conv3d':
        return 1
    elif nonlinearity == 'sigmoid':
        return 1
    elif nonlinearity == 'tanh':
        return 5/3
    elif nonlinearity == 'relu':
        return 2**0.5
    elif nonlinearity == 'leaky_relu':
        if param is None:
            raise ValueError("Negative slope must be provided for Leaky ReLU")
        return (2 / (1 + param**2))**0.5
    elif nonlinearity == 'selu':
        return 3/4
    else:
        raise ValueError("Unsupported nonlinearity")

if __name__ == "__main__":
    test_nonlinearities = ['linear', 'conv1d', 'sigmoid', 'tanh', 'relu', 'leaky_relu', 'selu']
    test_params = [None, 0.01, None]  # Only needed for Leaky ReLU

    for nl, param in zip(test_nonlinearities, test_params):
        gain = get_nonlinearity_gain(nl, param)
        print(f"Gain for {nl} with param {param if param is not None else 'None'} is {gain}")