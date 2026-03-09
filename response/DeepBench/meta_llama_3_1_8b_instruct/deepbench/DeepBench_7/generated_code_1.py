import nncf
from nncf.config import NNCFConfig
import torch
import torch.nn as nn

def create_compressed_model_and_algo_for_test(model, nncf_config=None, dummy_forward_func=None, wrap_inputs_func=None, compression_state=None):
    if nncf_config is not None:
        nncf_config.validate()
    
    compressed_model = nncf.compression.CompressionController.create_compressed_model(
        model, 
        compression_state, 
        dummy_forward_func=dummy_forward_func, 
        wrap_inputs_func=wrap_inputs_func
    )
    
    compression_algo_controller = nncf.compression.CompressionController(
        model, 
        nncf_config, 
        compression_state
    )
    
    return compressed_model, compression_algo_controller

if __name__ == "__main__":
    # Create a dummy neural network model
    class DummyModel(nn.Module):
        def __init__(self):
            super(DummyModel, self).__init__()
            self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
            self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
            self.fc1 = nn.Linear(320, 50)
            self.fc2 = nn.Linear(50, 10)
        
        def forward(self, x):
            x = torch.relu(torch.max_pool2d(self.conv1(x), 2))
            x = torch.relu(torch.max_pool2d(self.conv2(x), 2))
            x = x.view(-1, 320)
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    
    model = DummyModel()
    model.eval()
    
    # Create a dummy compression state
    compression_state = nncf.compression.CompressionState()
    
    # Create a dummy forward function
    def dummy_forward_func(model, input):
        return model(input)
    
    # Create a dummy wrap inputs function
    def wrap_inputs_func(input):
        return input
    
    # Create a dummy NNCF config
    nncf_config = NNCFConfig()
    
    # Call the function and print the results
    compressed_model, compression_algo_controller = create_compressed_model_and_algo_for_test(
        model, 
        nncf_config, 
        dummy_forward_func, 
        wrap_inputs_func, 
        compression_state
    )
    print("Compressed Model:", compressed_model)
    print("Compression Algorithm Controller:", compression_algo_controller)