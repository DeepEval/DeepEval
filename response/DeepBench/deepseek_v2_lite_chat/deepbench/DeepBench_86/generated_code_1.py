import torch
import torch.nn as nn
from warnings import warn

def function_name(args): 
    warn("Deprecated parameters n_hidden_units and n_hidden_layers are being replaced with hidden_units. Please update your usage.", DeprecationWarning, stacklevel=2)
    
    if args.get('hidden_units') is None:
        args['hidden_units'] = (32, 32)

    class FCNN(nn.Module):
        def __init__(self, n_input_units=1, n_output_units=1, actv=nn.Tanh, hidden_units=None):
            super().__init__()
            if hidden_units is None:
                hidden_units = args.get('hidden_units', (32, 32))
            layers = []
            num_layers = len(hidden_units)
            for i in range(num_layers - 1):
                layers.append(nn.Linear(hidden_units[i], hidden_units[i+1]))
                if i < num_layers - 2:
                    layers.append(actv())
            layers.append(nn.Linear(hidden_units[-1], n_output_units))
            self.NN = nn.Sequential(*layers)

        def forward(self, x):
            return self.NN(x)

    return FCNN(**args)

if __name__ == "__main__":
    import random
    n_input_units = 10
    n_output_units = 1
    n_hidden_units = (random.randint(1, 10), random.randint(1, 10))

    model = function_name(dict(n_input_units=n_input_units, n_output_units=n_output_units, n_hidden_units=n_hidden_units))
    print("Model structure:")
    print(model)
    print("Model parameters:", sum(p.numel() for p in model.parameters()))