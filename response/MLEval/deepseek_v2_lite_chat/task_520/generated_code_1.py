import torch
from torch import Tensor
from torch.sparse import SparseTensor

def convert_sparse_to_tensor(sparse_tensor: SparseTensor) -> Tensor:
    """
    Converts a PyTorch SparseTensor to a regular Tensor.

    Args:
    sparse_tensor (SparseTensor): The input SparseTensor to convert.

    Returns:
    Tensor: The converted Tensor.
    """
    # Convert the SparseTensor to a regular Tensor
    return sparse_tensor.to_dense()

def method():
    # Example SparseTensor
    indices = torch.tensor([[0, 0], [1, 1]])
    values = torch.tensor([1, 2])
    size = torch.Size([2, 2])
    sparse_tensor = SparseTensor(indices, size, values)

    # Convert SparseTensor to Tensor
    tensor = convert_sparse_to_tensor(sparse_tensor)
    
    return tensor

# Call the method for validation
output = method()
print("Output:", output)