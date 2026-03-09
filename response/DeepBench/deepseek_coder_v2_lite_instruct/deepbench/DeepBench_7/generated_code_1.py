# Import necessary packages
import torch
from nncf import create_compressed_model
from nncf.torch import NNCFConfig
from nncf.torch.utils import load_state

def create_compressed_model_and_algo_for_test(model, config=None, dummy_forward_fn=None, wrap_inputs_fn=None, compression_state_dict=None):
    # Validate the NNCFConfig object if provided
    if config:
        assert isinstance(config, NNCFConfig), "config must be an instance of NNCFConfig"
    
    # Create a compressed model
    compressed_model, compression_ctrl = create_compressed_model(model, config, dummy_forward_fn, wrap_inputs_fn, compression_state_dict)
    
    return compressed_model, compression_ctrl

if __name__ == "__main__":
    # Sample input values
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.fc = torch.nn.Linear(10, 1)

        def forward(self, x):
            return self.fc(x)

    model = SimpleModel()
    dummy_input = torch.randn(1, 10)

    # Call the function and print the results
    compressed_model, compression_ctrl = create_compressed_model_and_algo_for_test(model)
    print(compressed_model)
    print(compression_ctrl)