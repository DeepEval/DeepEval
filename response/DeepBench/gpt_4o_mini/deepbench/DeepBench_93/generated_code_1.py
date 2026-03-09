import torch
import torch.nn as nn

def build_layer(input_size, output_size, activation_function):
    layer_list = []
    layer_list.append(nn.Linear(input_size, output_size))
    
    if activation_function == 'relu':
        layer_list.append(nn.ReLU())
    elif activation_function == 'sigmoid':
        layer_list.append(nn.Sigmoid())
    elif activation_function == 'tanh':
        layer_list.append(nn.Tanh())
    elif activation_function == 'softmax':
        layer_list.append(nn.Softmax(dim=1))
        
    return nn.Sequential(*layer_list)

if __name__ == "__main__":
    input_tensor = torch.randn(1, 10)  # Sample input of size 10
    layer = build_layer(10, 5, 'relu')  # Build a layer with ReLU activation
    output = layer(input_tensor)  # Forward pass
    print(output)