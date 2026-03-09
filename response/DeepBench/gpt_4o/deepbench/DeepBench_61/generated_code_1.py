import torch
import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

def function_name(input_tensor, input_size, hidden_size, output_size): 
    model = SimpleMLP(input_size, hidden_size, output_size)
    output = model(input_tensor)
    return output

if __name__ == "__main__":
    input_tensor = torch.randn(1, 5)  # Sample input tensor with batch size of 1 and 5 features
    input_size = 5
    hidden_size = 10
    output_size = 2
    result = function_name(input_tensor, input_size, hidden_size, output_size)
    print("Output:", result)