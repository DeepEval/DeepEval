import math

def recommended_gain(nonlinearity, negative_slope=0.01):
    if nonlinearity in ['linear', 'identity', 'Conv1D', 'Conv2D', 'Conv3D', 'sigmoid']:
        return 1.0
    elif nonlinearity == 'tanh':
        return 5 / 3
    elif nonlinearity == 'relu':
        return math.sqrt(2)
    elif nonlinearity == 'leaky_relu':
        return math.sqrt(2 / (1 + negative_slope ** 2))
    elif nonlinearity == 'selu':
        return 3 / 4
    else:
        raise ValueError(f"Unknown nonlinearity: {nonlinearity}")

if __name__ == "__main__":
    print(recommended_gain('linear'))  # Output: 1.0
    print(recommended_gain('tanh'))    # Output: 1.6666666666666667
    print(recommended_gain('relu'))    # Output: 1.4142135623730951
    print(recommended_gain('leaky_relu', negative_slope=0.01))  # Output: 1.414213562373095
    print(recommended_gain('selu'))    # Output: 0.75