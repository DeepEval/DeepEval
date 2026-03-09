import torch
from torch.nn import Linear
from torch.optim import SGD
from nnc import NNCFConfig, CompressionAlgoController
from nnc.algorithms import (
    ActivationHinting,
    Pruning,
    Quantization,
)

def create_compressed_model_and_algo_for_test(
    model, nnc_config: NNCFConfig = None, dummy_forward=None, wrap_inputs=None, compression_state_dict=None
):
    if nnc_config is not None:
        nc_algo_ctrl = CompressionAlgoController(nnc_config)
    else:
        nc_algo_ctrl = None

    compressed_model = model
    if dummy_forward is not None:
        compressed_model = nc_algo_ctrl.wrap_forward(dummy_forward, compressed_model)
    if wrap_inputs is not None:
        compressed_model = nc_algo_ctrl.wrap_inputs(wrap_inputs, compressed_model)

    compressed_model.load_state_dict(compression_state_dict)
    return compressed_model, nc_algo_ctrl

# Example usage
if __name__ == "__main__":
    # Sample input values
    input_tensor = torch.randn(1, 3, 28, 28)

    # Define the model
    model = Linear(784, 10)

    # Create a NNCFConfig object (optional)
    nnc_config = NNCFConfig(
        {
            "pruning": {"magnitude": 0.5},
            "quantization": {"dtype": torch.qint8},
            "activation_hinting": {"dtype": torch.float16},
        }
    )

    # Create the compressed model and algorithm controller
    compressed_model, nc_algo_ctrl = create_compressed_model_and_algo_for_test(
        model, nnc_config, dummy_forward=model(input_tensor), wrap_inputs=lambda x: x.view(-1, 784), compression_state_dict=model.state_dict()
    )

    # Print the compressed model and algorithm controller
    print("Compressed Model:", compressed_model)
    print("Compression Algorithm Controller:", nc_algo_ctrl)