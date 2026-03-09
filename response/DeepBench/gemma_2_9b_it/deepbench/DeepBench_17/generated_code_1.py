import torch
from numba import jit
from numba.cuda import cuda
from nNCF import NNCFNetwork, nncf_model_input, wrap_inputs_fn


def create_nncf_network(model, config, dummy_forward_fn=None, wrap_inputs_fn=None, wrap_outputs_fn=None):
    if dummy_forward_fn is not None:
        assert wrap_inputs_fn is not None, "wrap_inputs_fn must be specified if dummy_forward_fn is provided"
        model = NNCFNetwork(model, config, dummy_forward_fn=dummy_forward_fn, wrap_inputs_fn=wrap_inputs_fn, wrap_outputs_fn=wrap_outputs_fn)
    else:
        model = NNCFNetwork(model, config)
    return model

if __name__ == "__main__":
    dummy_input = torch.randn(1, 3, 224, 224)
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 16, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=2, stride=2),
        torch.nn.Conv2d(16, 32, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=2, stride=2),
        torch.nn.Flatten(),
        torch.nn.Linear(32 * 56 * 56, 10)
    )

    config = {'compression_algorithm': 'quantization', 'quantization_config': {'qscheme': 'qint8'}}
    wrapped_model = create_nncf_network(model, config)
    output = wrapped_model(dummy_input)
    print(output.shape)