import torch
import torch.nn as nn
from warnings import warn

class FCNN(nn.Module):
    def __init__(self, n_input_units=1, n_output_units=1, n_hidden_units=None, n_hidden_layers=None, actv=nn.Tanh, hidden_units=None):
        super(FCNN, self).__init__()
        
        if n_hidden_units is not None or n_hidden_layers is not None:
            warn("The parameters n_hidden_units and n_hidden_layers are deprecated. Please use hidden_units instead.")
        
        if hidden_units is None:
            hidden_units = (32, 32)
        
        layers = []
        layer_sizes = [n_input_units] + list(hidden_units) + [n_output_units]
        
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                layers.append(actv())
        
        self.NN = nn.Sequential(*layers)

def function_name(args): 
    # Your code here

    return output

if __name__ == "__main__":
    # Your runnable example here
    fcnn = FCNN(n_input_units=4, n_output_units=2, hidden_units=(16, 8))
    sample_input = torch.randn(1, 4)
    output = fcnn(sample_input)
    print(output)