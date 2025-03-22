import torch
import torch.nn as nn

def method():
    # Simulating the output from a final fully connected layer (e.g., logits)
    logits = torch.tensor([[2.0, 1.0, 0.1]])

    # Applying the softmax function
    softmax = nn.Softmax(dim=1)
    output = softmax(logits)

    return output

# Call the method() for validation
output = method()
print(output)