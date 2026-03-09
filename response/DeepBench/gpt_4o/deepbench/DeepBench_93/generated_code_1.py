import torch
import torch.nn as nn

def build_layer(input_dim, hidden_dims, output_dim):
    layers = []
    in_dim = input_dim
    for h_dim in hidden_dims:
        layers.append(nn.Linear(in_dim, h_dim))
        layers.append(nn.ReLU())
        in_dim = h_dim
    layers.append(nn.Linear(in_dim, output_dim))
    return nn.Sequential(*layers)

if __name__ == "__main__":
    # Example usage of the build_layer function
    input_dim = 10
    hidden_dims = [20, 30, 20]
    output_dim = 5
    model = build_layer(input_dim, hidden_dims, output_dim)
    
    # Creating a sample input tensor
    sample_input = torch.randn(1, input_dim)  # batch size of 1, input_dim features

    # Passing the sample input through the model
    output = model(sample_input)

    # Printing the output
    print("Output of the MLP model:", output)