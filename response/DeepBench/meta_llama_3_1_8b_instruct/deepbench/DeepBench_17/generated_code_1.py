import torch
from nncf.api.compression import NNCFNetwork
from nncf.api.compression import CompressionConfig

def create_nncf_network(model, config, dummy_forward_fn=None, wrap_inputs_fn=None, wrap_outputs_fn=None):
    if dummy_forward_fn and not wrap_inputs_fn:
        raise ValueError("wrap_inputs_fn should be specified when dummy_forward_fn is provided")
    
    if dummy_forward_fn:
        wrapped_inputs = wrap_inputs_fn(model.forward, model.__kwdefaults__)
        dummy_forward_fn(*wrapped_inputs)
    else:
        model.forward(torch.randn(1, 3, 224, 224))
    
    nncf_network = NNCFNetwork(model, config)
    
    return nncf_network

def wrap_inputs_fn(forward_call, kwargs):
    def wrapper(*args, **kwargs):
        wrapped_args = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                wrapped_args.append(nncf_model_input(arg))
            else:
                wrapped_args.append(arg)
        
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                kwargs[key] = nncf_model_input(value)
        
        return forward_call(*wrapped_args, **kwargs)
    
    return wrapper

def wrap_outputs_fn(forward_call, kwargs):
    def wrapper(*args, **kwargs):
        outputs = forward_call(*args, **kwargs)
        wrapped_outputs = []
        for output in outputs:
            if isinstance(output, torch.Tensor):
                wrapped_outputs.append(nncf_model_input(output))
            else:
                wrapped_outputs.append(output)
        
        return wrapped_outputs
    
    return wrapper

def nncf_model_input(tensor):
    return tensor

if __name__ == "__main__":
    # Create a sample model
    class SampleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
            self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
            self.fc1 = torch.nn.Linear(320, 50)
            self.fc2 = torch.nn.Linear(50, 10)
        
        def forward(self, x):
            x = torch.relu(torch.nn.functional.avg_pool2d(self.conv1(x), 2))
            x = torch.relu(torch.nn.functional.avg_pool2d(self.conv2(x), 2))
            x = x.view(-1, 320)
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    
    model = SampleModel()
    
    # Create a sample compression config
    config = CompressionConfig()
    
    # Create a sample dummy forward function
    def dummy_forward_fn():
        model.forward(torch.randn(1, 1, 28, 28))
    
    # Create sample input values
    dummy_forward_fn = dummy_forward_fn
    
    wrap_inputs_fn = wrap_inputs_fn(model.forward, model.__kwdefaults__)
    wrap_outputs_fn = wrap_outputs_fn(model.forward, model.__kwdefaults__)
    
    # Call the function and print the results
    nncf_network = create_nncf_network(model, config, dummy_forward_fn, wrap_inputs_fn, wrap_outputs_fn)
    print(nncf_network)