import numpy as np
import torch

def aepe(input, target, reduction='mean'):
    input = torch.tensor(input)
    target = torch.tensor(target)
    assert input.shape == target.shape, "Input and target shapes must be the same"
    assert input.shape[-1] == 2, "Input must have shape (*, 2)"

    diff = input - target
    sq_diff = torch.sum(diff ** 2, dim=-1)
    aepe = torch.sqrt(sq_diff)
    aepe = torch.mean(aepe.view(-1))

    if reduction == 'none':
        return aepe
    elif reduction =='mean':
        return aepe
    elif reduction =='sum':
        return torch.sum(aepe)
    else:
        raise ValueError("Invalid reduction")

if __name__ == "__main__":
    # Create sample input values
    input_flow = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    target_flow = np.array([[[2, 3], [4, 5]], [[6, 7], [8, 9]]])

    # Call the function
    result1 = aepe(input_flow, target_flow, reduction='mean')
    result2 = aepe(input_flow, target_flow, reduction='sum')
    result3 = aepe(input_flow, target_flow, reduction='none')

    # Print the results
    print("AEPE with mean reduction: ", result1)
    print("AEPE with sum reduction: ", result2)
    print("AEPE with none reduction: ", result3)