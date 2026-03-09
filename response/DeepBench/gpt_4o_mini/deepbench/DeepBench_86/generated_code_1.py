import torch
import torch.nn as nn
from warnings import warn

class FCNN(nn.Module):
    def __init__(self, n_input_units=1, n_output_units=1, n_hidden_units=None, n_hidden_layers=None, actv=nn.Tanh, hidden_units=None):
        super(FCNN, self).__init__()
        
        if n_hidden_units is not None or n_hidden_layers is not None:
            warn("n_hidden_units and n_hidden_layers are deprecated. Use hidden_units instead.")
        
        if hidden_units is None:
            hidden_units = (32, 32)
        
        layers = []
        input_size = n_input_units
        
        for units in hidden_units:
            layers.append(nn.Linear(input_size, units))
            layers.append(actv())
            input_size = units
        
        layers.append(nn.Linear(input_size, n_output_units))
        
        self.NN = nn.Sequential(*layers)

    def forward(self, x):
        return self.NN(x)

if __name__ == "__main__":
    model = FCNN(n_input_units=3, n_output_units=1)
    sample_input = torch.randn(5, 3)  # Batch size of 5 and input size of 3
    output = model(sample_input)
    print(output)