import torch.nn as nn

def get_nonlinearity_gain(nonlinearity, param=None):
    nonlinearity_dict = {
        'linear': 1,
        'conv1d': 1,
        'conv2d': 1,
        'conv3d': 1,
        'sigmoid': 1,
        'tanh': 5 / 3,
        'relu': (2 ** 0.5),
        'leaky_relu': (2 / (1 + (param ** 2))) ** 0.5,
        'selu': 3 / 4
    }

    return nonlinearity_dict.get(nonlinearity, 1)


if __name__ == "__main__":
    nonlinearity = 'selu'
    param = 0.2

    gain = get_nonlinearity_gain(nonlinearity, param)

    print(gain)