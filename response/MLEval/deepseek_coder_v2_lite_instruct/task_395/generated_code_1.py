import torch

def method():
    # Create two tensors
    tensor1 = torch.tensor([1, 2, 3])
    tensor2 = torch.tensor([4, 5, 6])

    # Perform element-wise addition
    output = tensor1 + tensor2

    return output

# Call the method and print the result
result = method()
print(result)