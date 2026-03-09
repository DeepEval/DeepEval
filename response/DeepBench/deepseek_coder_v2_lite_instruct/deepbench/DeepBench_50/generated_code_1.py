import torch

def aepe(input, target, reduction='mean'):
    error = torch.sqrt(((input[:, 0] - target[:, 0]) ** 2 + (input[:, 1] - target[:, 1]) ** 2))
    if reduction == 'none':
        return error
    elif reduction == 'mean':
        return torch.mean(error)
    elif reduction == 'sum':
        return torch.sum(error)

if __name__ == "__main__":
    # Sample input values
    input_flow_map = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    target_flow_map = torch.tensor([[[1.1, 2.1], [3.1, 4.1]], [[5.1, 6.1], [7.1, 8.1]]])

    # Call the function and print the results
    result = aepe(input_flow_map, target_flow_map, reduction='mean')
    print(f"AEPE: {result.item()}")