import torch
import torch.nn as nn
from nncf import NNCFConfig
from nncf.torch import create_compressed_model, NNCFNetwork

def create_nncf_network(model: nn.Module, config: NNCFConfig, dummy_forward_fn=None, wrap_inputs_fn=None, wrap_outputs_fn=None):
    if dummy_forward_fn and not wrap_inputs_fn:
        raise ValueError("If dummy_forward_fn is specified, wrap_inputs_fn must also be specified.")

    def default_dummy_forward_fn(model):
        # Use mock tensors based on the input size specified in config
        input_sizes = config.get("input_info", [{"shape": [1, 3, 224, 224]}])
        mock_inputs = [torch.randn(size["shape"]) for size in input_sizes]
        return model(*mock_inputs)

    dummy_forward_fn = dummy_forward_fn or default_dummy_forward_fn

    nncf_network = create_compressed_model(
        model=model,
        config=config,
        dummy_forward_fn=dummy_forward_fn,
        wrap_inputs_fn=wrap_inputs_fn,
        wrap_outputs_fn=wrap_outputs_fn
    )
    
    return nncf_network

if __name__ == "__main__":
    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.conv = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
            self.relu = nn.ReLU()
            self.fc = nn.Linear(16 * 224 * 224, 10)

        def forward(self, x):
            x = self.conv(x)
            x = self.relu(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)

    model = SimpleModel()
    
    config_dict = {
        "input_info": [{"shape": [1, 3, 224, 224]}],
        "compression": [{"algorithm": "quantization"}]
    }
    config = NNCFConfig.from_dict(config_dict)

    nncf_model = create_nncf_network(model, config)

    dummy_input = torch.randn(1, 3, 224, 224)
    output = nncf_model(dummy_input)
    print(output)