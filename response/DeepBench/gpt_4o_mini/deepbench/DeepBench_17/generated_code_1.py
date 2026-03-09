import torch
import torch.nn as nn
from nncf import NNCFNetwork
from nncf import nncf_model_input

def create_nncf_network(model, config, dummy_forward_fn=None, wrap_inputs_fn=None, wrap_outputs_fn=None):
    if dummy_forward_fn is not None:
        dummy_input = (torch.randn(1, 3, 224, 224),)  # Example input shape for an image
        dummy_forward_fn(*dummy_input)
    
    def wrap_inputs(args, kwargs):
        new_args = tuple(nncf_model_input(arg) if isinstance(arg, torch.Tensor) else arg for arg in args)
        new_kwargs = {k: nncf_model_input(v) if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}
        return new_args, new_kwargs
    
    def wrap_outputs(output):
        return nncf_model_input(output) if isinstance(output, torch.Tensor) else output

    model = NNCFNetwork(model, config)
    model.wrap_forward(wrap_inputs, wrap_outputs)
    
    return model

class SampleModel(nn.Module):
    def forward(self, x):
        return x * 2

if __name__ == "__main__":
    model = SampleModel()
    config = {}  # Placeholder for the configuration object
    nncf_model = create_nncf_network(model, config)
    
    sample_input = torch.randn(1, 3, 224, 224)
    output = nncf_model(sample_input)
    print(output)