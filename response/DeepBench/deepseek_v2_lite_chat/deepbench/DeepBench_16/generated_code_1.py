import torch
from typing import Optional, Tuple, List
from torch import Tensor, device

class Transform3d:
    def __init__(self, dtype: torch.dtype = torch.float32, device: Device = "cpu", matrix: Optional[Tensor] = None):
        if matrix is None:
            self._matrix = torch.eye(4, device=device, dtype=dtype)
        else:
            self._matrix = matrix
        self._transforms = []
        self.device = device
        self.dtype = dtype

    def append(self, transform: 'Transform3d'):
        if isinstance(transform, Transform3d):
            self._transforms.append(transform)
        else:
            raise ValueError("Can only append another Transform3d instance.")

    def get_matrix(self) -> Tensor:
        current_matrix = self._matrix
        for transform in reversed(self._transforms):
            current_matrix = transform.matrix @ current_matrix
        return current_matrix

def broadcast_transforms(transforms: List[Tensor]) -> Tensor:
    if len(transforms) == 1:
        return transforms[0]
    else:
        return torch.stack([t.unsqueeze(0) for t in transforms]).mean(0)

def function_name(args): 
    # Initialize the first transform
    transform1 = Transform3d(dtype=torch.float32, device="cpu")
    transform1.matrix = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], device=transform1.device)
    
    # Initialize additional transforms
    transforms_batch = [transform1] * 4

    # Append transforms to the batch
    for i in range(4):
        transforms_batch[i].append(Transform3d(dtype=torch.float32, device=transforms_batch[i].device))
        transforms_batch[i].matrix = torch.tensor([[0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]], device=transforms_batch[i].device)

    # Get the composite matrix
    composite_matrix = broadcast_broadcast_transforms(transforms_batch)
    print("Composite Matrix:\n", composite_matrix)

    return composite_matrix

if __name__ == "__main__":
    composite_matrix = function_name(None)