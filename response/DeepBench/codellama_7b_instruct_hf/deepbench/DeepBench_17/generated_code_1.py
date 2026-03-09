import torch
import nncf

def create_nncf_network(model, config, dummy_forward_fn=None, wrap_inputs_fn=None):
    # Create a dummy input tensor if not provided
    if dummy_forward_fn is None:
        dummy_input = torch.randn(1, 3, 224, 224)
    else:
        dummy_input = None

    # Create an NNCFNetwork wrapper
    nncf_network = nncf.NNCFNetwork(model, dummy_input)

    # Set the compression config
    nncf_network.set_compression_config(config)

    # Wrap the input tensors with nncf_model_input if specified
    if wrap_inputs_fn is not None:
        nncf_network.wrap_inputs_fn = wrap_inputs_fn

    return nncf_network

if __name__ == "__main__":
    import torch.nn as nn
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, 3)
            self.relu = nn.ReLU()
        def forward(self, x):
            return self.relu(self.conv(x))
    model = SimpleModel()
    config = nncf.CompressionConfig()
    nncf_network = create_nncf_network(model, config)
    test_input = torch.randn(1, 3, 224, 224)
    output = nncf_network(test_input)
    print("Output shape:", output.shape)