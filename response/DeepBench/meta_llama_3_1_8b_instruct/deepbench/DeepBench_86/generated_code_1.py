import torch
import torch.nn as nn
from warnings import warn

class FCNN(nn.Module):
    def __init__(self, n_input_units=1, n_output_units=1, n_hidden_units=None,
  n_hidden_layers=None, actv=nn.Tanh, hidden_units=None):
        super(FCNN, self).__init__()
        
        # Handle deprecated parameters
        if n_hidden_units is not None:
            warn("The parameter 'n_hidden_units' is deprecated, use 'hidden_units'
  instead.", DeprecationWarning)
        if n_hidden_layers is not None:
            warn("The parameter 'n_hidden_layers' is deprecated, use 'hidden_units'
  instead.", DeprecationWarning)
        
        # Default hidden units if not provided
        if hidden_units is None:
            hidden_units = (32, 32)
        
        # Initialize the network
        layers = [nn.Linear(n_input_units, hidden_units[0]), actv()]
        for i in range(len(hidden_units) - 1):
            layers.append(nn.Linear(hidden_units[i], hidden_units[i+1]))
            layers.append(actv())
        layers.append(nn.Linear(hidden_units[-1], n_output_units))
        
        # Remove the last activation function
        layers.pop()
        
        self.NN = nn.Sequential(*layers)

if __name__ == "__main__":
    # Create a sample FCNN
    fcnn = FCNN(n_input_units=10, n_output_units=5, hidden_units=(64, 32))
    
    # Create a sample input
    x = torch.randn(1, 10)
    
    # Forward pass
    output = fcnn.NN(x)
    
    # Print the output shape
    print(output.shape)