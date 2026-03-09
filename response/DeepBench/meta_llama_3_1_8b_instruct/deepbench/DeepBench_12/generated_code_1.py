import torch
import torch.nn.functional as F

def get_gain(nonlinearity, param=None):
    if nonlinearity == 'linear':
        return 1
    elif nonlinearity == 'conv1d' or nonlinearity == 'conv2d' or nonlinearity
  == 'conv3d':
        return 1
    elif nonlinearity =='sigmoid':
        return 1
    elif nonlinearity == 'tanh':
        return 5/3
    elif nonlinearity =='relu':
        return (2)**0.5
    elif nonlinearity == 'leaky_relu':
        return (2/(1 + param**2))**0.5
    elif nonlinearity =='selu':
        return 3/4
    else:
        raise ValueError('Unsupported nonlinearity')

if __name__ == "__main__":
    # Create sample input values
    nonlinearities = ['linear','sigmoid', 'tanh','relu', 'leaky_relu','selu']
    params = [None, None, None, 0.1, 0.1, None]
    

    for nonlinearity, param in zip(nonlinearities, params):
        gain = get_gain(nonlinearity, param)
        print(f'Nonlinearity: {nonlinearity}, Gain: {gain}')