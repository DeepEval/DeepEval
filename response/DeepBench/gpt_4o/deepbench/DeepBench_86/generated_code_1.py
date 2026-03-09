import torch
import torch.nn as nn
from warnings import warn

class FCNN(nn.Module):
    def __init__(self, n_input_units=1, n_output_units=1, n_hidden_units=None, n_hidden_layers=None, actv=nn.Tanh, hidden_units=None):
        super(FCNN, self).__init__()
        
        # Handle deprecated parameters
        if n_hidden_units is not None or n_hidden_layers is not None:
            warn("n_hidden_units and n_hidden_layers are deprecated. Use hidden_units instead.", DeprecationWarning)
            if hidden_units is None:
                if n_hidden_layers is None:
                    n_hidden_layers = 2  # Default number of hidden layers
                if n_hidden_units is None:
                    n_hidden_units = 32  # Default number of units in each hidden layer
                hidden_units = (n_hidden_units,) * n_hidden_layers
        
        # Default hidden units if not provided
        if hidden_units is None:
            hidden_units = (32, 32)
        
        # Build the network layers
        layers = []
        input_dim = n_input_units
        for units in hidden_units:
            layers.append(nn.Linear(input_dim, units))
            layers.append(actv())
            input_dim = units
        
        # Output layer (no activation)
        layers.append(nn.Linear(input_dim, n_output_units))
        
        # Register layers as a module
        self.NN = nn.Sequential(*layers)

    def forward(self, x):
        return self.NN(x)

if __name__ == "__main__":
    # Create a sample FCNN
    model = FCNN(n_input_units=4, n_output_units=2, hidden_units=(64, 128))
    
    # Generate some random input data
    sample_input = torch.randn(10, 4)
    
    # Forward pass through the model
    output = model(sample_input)
    
    # Print the model output
    print(output)