import torch
import nncf

class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = torch.nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

def create_nncf_network(model, config, dummy_forward_fn=None, wrap_inputs_fn=None, wrap_outputs_fn=None):
    from nncf import NNCFConfig, NNCFNetwork

    if dummy_forward_fn and wrap_inputs_fn:
        raise ValueError("Both dummy_forward_fn and wrap_inputs_fn cannot be specified simultaneously.")

    if dummy_forward_fn:
        def dummy_input_fn(input_shape):
            return [torch.randn(input_shape)]

        def dummy_forward(wrapped_model, *args, **kwargs):
            for arg in args:
                nncf.nncf_model_input(arg)
            for key, value in kwargs.items():
                nncf.nncf_model_input(value)
            return wrapped_model(*args, **kwargs)

        dummy_forward_fn = dummy_forward
        wrap_inputs_fn = lambda args, kwargs: args, kwargs

    nncf_config = NNCFConfig.from_dict(config)
    wrapped_model = NNCFNetwork(model, nncf_config)

    if dummy_forward_fn:
        dummy_input = dummy_input_fn(wrapped_model.input_shape)
        dummy_forward(wrapped_model, *dummy_input)

    return wrapped_model

if __name__ == "__main__":
    model = SimpleModel()
    config = {
        "input_info": {
            "sample_size": [1, 10]
        }
    }
    dummy_input_shape = [1, 10]
    wrapped_model = create_nncf_network(model, config, dummy_forward_fn=None, wrap_inputs_fn=None)
    print(wrapped_model)