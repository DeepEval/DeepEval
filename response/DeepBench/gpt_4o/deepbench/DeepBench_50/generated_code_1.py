import numpy as np

def aepe(input, target, reduction='mean'):
    assert input.shape == target.shape, "Input and target must have the same shape."
    assert input.shape[-1] == 2, "The last dimension must be 2 for 2D vectors."

    diff = input - target
    squared_diff = np.square(diff)
    sum_squared_diff = np.sum(squared_diff, axis=-1)
    endpoint_error = np.sqrt(sum_squared_diff)

    if reduction == 'none':
        return endpoint_error
    elif reduction == 'mean':
        return np.mean(endpoint_error)
    elif reduction == 'sum':
        return np.sum(endpoint_error)
    else:
        raise ValueError("Reduction must be 'none', 'mean', or 'sum'.")

if __name__ == "__main__":
    input_flow = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    target_flow = np.array([[[1.5, 2.5], [3.5, 4.5]], [[4.5, 5.5], [6.5, 7.5]]])
    
    aepe_none = aepe(input_flow, target_flow, reduction='none')
    aepe_mean = aepe(input_flow, target_flow, reduction='mean')
    aepe_sum = aepe(input_flow, target_flow, reduction='sum')
    
    print("AEPE with no reduction:", aepe_none)
    print("AEPE with mean reduction:", aepe_mean)
    print("AEPE with sum reduction:", aepe_sum)