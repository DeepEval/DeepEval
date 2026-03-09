import torch
import torch.nn as nn
from nncf import NNCFConfig
from nncf import create_compressed_model
from nncf.torch import create_compression_algorithm
from nncf.torch import create_nncf_model

def create_compressed_model_and_algo_for_test(model, nncf_config=None, dummy_forward_fn=None, wrap_inputs_fn=None, compression_state_dict=None):
    if nncf_config is not None:
        if not isinstance(nncf_config, NNCFConfig):
            raise ValueError("Provided NNCFConfig object is invalid.")
        
    compressed_model, compression_ctrl = create_compressed_model(model, nncf_config)

    if compression_state_dict:
        compression_ctrl.load_state(compression_state_dict)

    return compressed_model, compression_ctrl

if __name__ == "__main__":
    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.conv = nn.Conv2d(1, 1, 3)
            self.relu = nn.ReLU()

        def forward(self, x):
            return self.relu(self.conv(x))

    model = SimpleModel()

    nncf_config_dict = {
        "input_info": {
            "sample_size": [1, 1, 5, 5],
        },
        "compression": [
            {
                "algorithm": "quantization"
            }
        ]
    }
    nncf_config = NNCFConfig.from_dict(nncf_config_dict)

    def dummy_forward_fn(model):
        return model(torch.rand(1, 1, 5, 5))

    def wrap_inputs_fn(*args, **kwargs):
        return args, kwargs

    compressed_model, compression_ctrl = create_compressed_model_and_algo_for_test(
        model, nncf_config, dummy_forward_fn, wrap_inputs_fn
    )

    print("Compressed Model:", compressed_model)
    print("Compression Controller:", compression_ctrl)