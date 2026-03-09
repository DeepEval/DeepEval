import torch
from nncf import NNCFConfig
from nncf.torch.compression_method_api import CompressionAlgorithmController

def create_compressed_model_and_algo_for_test(model, nncf_config=None, dummy_forward_fn=None, wrap_inputs_fn=None, compression_state_dict=None):
    # Validate the NNCFConfig object if provided
    if nncf_config is not None:
        assert isinstance(nncf_config, NNCFConfig)

    # Create a compressed model using the input parameters
    compressed_model = model
    if nncf_config is not None:
        compressed_model = nncf_config.get_compression_model(model)

    # Create a compression algorithm controller
    algo_controller = CompressionAlgorithmController(compressed_model, dummy_forward_fn, wrap_inputs_fn, compression_state_dict)

    return compressed_model, algo_controller

# Example usage:
if __name__ == "__main__":
    # Create a sample model
    model = torch.nn.Linear(5, 3)

    # Create a sample NNCFConfig object
    nncf_config = NNCFConfig()

    # Create a sample dummy forward function
    def dummy_forward_fn(inputs):
        return model(inputs)

    # Create a sample wrap inputs function
    def wrap_inputs_fn(inputs):
        return inputs

    # Create a sample compression state dictionary
    compression_state_dict = {"some_key": "some_value"}

    # Call the function and print the results
    compressed_model, algo_controller = create_compressed_model_and_algo_for_test(model, nncf_config, dummy_forward_fn, wrap_inputs_fn, compression_state_dict)
    print(compressed_model)
    print(algo_controller)