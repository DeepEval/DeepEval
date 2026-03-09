import torch

def get_gain(nonlinearity, param=None):
    if nonlinearity == 'linear' or nonlinearity == 'conv1d' or nonlinearity == 'conv2d' or nonlinearity == 'conv3d':
        return 1
    elif nonlinearity == 'sigmoid':
        return 1
    elif nonlinearity == 'tanh':
        return 5/3
    elif nonlinearity == 'relu':
        return torch.sqrt(2)
    elif nonlinearity == 'leaky_relu':
        return torch.sqrt(2 / (1 + param**2))
    elif nonlinearity == 'selu':
        return 3/4
    else:
        raise ValueError(f"Unsupported nonlinearity: {nonlinearity}")

if __name__ == "__main__":
    nonlinearities = ['relu', 'leaky_relu', 'selu', 'linear', 'sigmoid', 'tanh']
    for nonlinearity in nonlinearities:
        if nonlinearity == 'leaky_relu':
            param = 0.01
        else:
            param = None
        gain = get_gain(nonlinearity, param)
        print(f"Nonlinearity: {nonlinearity}, Gain: {gain}")