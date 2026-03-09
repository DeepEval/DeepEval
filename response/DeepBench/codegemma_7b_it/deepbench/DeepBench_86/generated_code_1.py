from torch import nn

class FCNN(nn.Module):
    def __init__(self, n_input_units=1, n_output_units=1, n_hidden_units=None,
  n_hidden_layers=None, actv=nn.Tanh, hidden_units=None):
        super().__init__()

        if n_hidden_units is not None:
            warn(
                "The ''n_hidden_units'' and ''n_hidden_layers'' arguments are deprecated."
                "Please use ''hidden_units'' instead.",
                DeprecationWarning,
            )
            if hidden_units is None:
                hidden_units = (n_hidden_units,) * n_hidden_layers

        if hidden_units is None:
            hidden_units = (32, 32)

        layers = []
        for n_units in hidden_units:
            layers.append(nn.Linear(n_input_units, n_units))
            layers.append(actv())
            n_input_units = n_units
        layers.append(nn.Linear(n_input_units, n_output_units))
        self.NN = nn.Sequential(*layers)

if __name__ == "__main__":
    # Example test case
    model = FCNN(n_input_units=2, n_output_units=1)
    print(model.NN)