import torch
from nncf import NNCF_Trainer, NNCF_Config
from torch import nn

def function_name(model, config, dummy_forward_fn=None, wrap_inputs_fn=None, wrap_outputs_fn=None):
    # Prepare NNCF model
    nncf_trainer = NNCF_Trainer(model)
    nncf_config = NNCF_Config(**config.dict())

    # Add compression
    nncf_trainer.add_model(model, nncf_config)

    # Check if dummy_forward_fn is provided and wrap inputs
    if dummy_forward_fn:
        model.dummy_forward_fn = dummy_forward_fn
    elif wrap_inputs_fn:
        model = nncf_trainer.wrap_inputs(model, wrap_inputs_fn)

    # Check if wrap_outputs_fn is provided and wrap outputs
    if wrap_outputs_fn:
        model = nncf_trainer.wrap_outputs(model, wrap_outputs_fn)

    # Finalize NNCF model
    nncf_model = nncf_trainer.finalize()

    return nncf_model

if __name__ == "__main__":
    # Sample configuration
    config = {
        "compression": {
            "method": "quantization",
            "params": {
                "scale": 127,
                "zero_point": 127,
                "output_layer": "classifier"
            }
        }
    }

    # Sample model (e.g., a simple model with a forward method)
    class SampleModel(nn.Module):
        def __init__(self):
            super(SampleModel, self).__init__()
            self.fc = nn.Linear(10, 5)

        def forward(self, x):
            return self.fc(x)

    model = SampleModel()

    # Call the function with the model and configuration
    nncf_model = function_name(model, config)

    # Create input for the model
    input_data = torch.randn(1, 10)

    # Forward pass
    output = nncf_model(input_data)

    # Print output
    print("Output from compressed model:", output)