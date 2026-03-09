import numpy as np
import torch
import torch.nn as nn

def build_layer(layer_size, num_layers, input_size):
    layer_list = []
    for i in range(num_layers):
        if i == 0:
            layer = nn.Linear(input_size, layer_size)
        else:
            layer = nn.Linear(layer_size, layer_size)
        layer_list.append(layer)
        layer_list.append(nn.ReLU())
    layer_list.append(nn.Linear(layer_size, 1))
    return layer_list

if __name__ == "__main__":
    input_size = 784
    num_layers = 3
    layer_size = 256
    layer_list = build_layer(layer_size, num_layers, input_size)
    
    # Convert layer_list to a neural network
    model = nn.Sequential(*layer_list)
    
    # Print the model
    print(model)
    
    # Create some sample input values
    input_values = torch.randn(1, input_size)
    
    # Print the output of the model for the input values
    output = model(input_values)
    print(output)