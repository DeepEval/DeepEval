import torch
import torch.nn as nn
from nncf import NNCFConfig
from nncf.torch import create_compressed_model_and_algo_for_test

def create_compressed_model_and_algo_for_test(model, nncf_config=None, dummy_forward=None, wrap_inputs=None, compression_state=None):
    if nncf_config is not None:
        if not isinstance(nncf_config, NNCFConfig):
            raise ValueError("Provided nncf_config is not of type NNCFConfig.")
    
    compressed_model, compression_algo = create_compressed_model_and_algo_for_test(model, nncf_config, dummy_forward, wrap_inputs, compression_state)
    
    return compressed_model, compression_algo

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)

def dummy_forward_fn(model, inputs):
    return model(inputs)

def wrap_inputs_fn(inputs):
    return inputs

if __name__ == "__main__":
    model = SimpleModel()
    nncf_config = NNCFConfig.from_json('path_to_config.json')  # Placeholder, adjust as needed
    dummy_inputs = torch.randn(1, 10)

    compressed_model, compression_algo = create_compressed_model_and_algo_for_test(
        model,
        nncf_config=nncf_config,
        dummy_forward=dummy_forward_fn,
        wrap_inputs=wrap_inputs_fn,
        compression_state=None
    )
    
    print(compressed_model)
    print(compression_algo)