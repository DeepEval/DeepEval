import math

def recommended_gain(nonlinearity, param=None):
    if nonlinearity in ['linear', 'identity', 'conv1d', 'conv2d', 'conv3d', 'sigmoid']:
        return 1
    elif nonlinearity == 'tanh':
        return 5 / 3
    elif nonlinearity == 'relu':
        return math.sqrt(2)
    elif nonlinearity == 'leaky_relu':
        if param is not None:
            negative_slope = param
        else:
            negative_slope = 0.01  # default value commonly used
        return math.sqrt(2 / (1 + negative_slope ** 2))
    elif nonlinearity == 'selu':
        return 3 / 4
    else:
        raise ValueError(f"Unsupported nonlinearity: {nonlinearity}")

if __name__ == "__main__":
    # Example usage
    nonlinearities = ['linear', 'tanh', 'relu', 'leaky_relu', 'selu']
    for nonlinearity in nonlinearities:
        gain = recommended_gain(nonlinearity)
        print(f"Gain for {nonlinearity}: {gain}")

    # Example with leaky_relu with specific negative_slope
    gain = recommended_gain('leaky_relu', param=0.2)
    print(f"Gain for leaky_relu with negative_slope=0.2: {gain}")