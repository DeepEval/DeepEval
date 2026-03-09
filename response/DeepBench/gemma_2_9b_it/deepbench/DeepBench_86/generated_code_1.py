import torch
import torch.nn as nn
from warnings import warn

class FCNN(nn.Module):
    def __init__(self, n_input_units=1, n_output_units=1, n_hidden_units=None,
  n_hidden_layers=None, actv=nn.Tanh, hidden_units=None):
        super(FCNN, self).__init__()
        if n_hidden_units is not None and n_hidden_layers is not None:
            warn("Parameters n_hidden_units and n_hidden_layers are deprecated.
  Please use hidden_units instead.")
            hidden_units = (n_hidden_units,) * n_hidden_layers
        if hidden_units is None:
            hidden_units = (32, 32)
        layers = [nn.Linear(n_input_units, hidden_units[0])]
        for i in range(len(hidden_units) - 1):
            layers.append(nn.Linear(hidden_units[i], hidden_units[i + 1]))
            layers.append(actv())
        layers.append(nn.Linear(hidden_units[-1], n_output_units))
        self.NN = nn.Sequential(*layers)

if __name__ == "__main__":
    input_values = torch.randn(1, 1)
    model = FCNN()
    output = model(input_values)
    print(output)