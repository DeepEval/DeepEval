import numpy as np

def aepe(input, target, reduction='mean'):
    assert input.shape == target.shape, "Input and target must have the same shape."
    
    h, w, _ = input.shape
    endpoint_errors = np.sqrt((input[..., 0] - target[..., 0]) ** 2 + (input[..., 1] - target[..., 1]) ** 2)
    
    if reduction == 'none':
        return endpoint_errors
    elif reduction == 'mean':
        return np.sum(endpoint_errors) / (h * w)
    elif reduction == 'sum':
        return np.sum(endpoint_errors)
    else:
        raise ValueError("Reduction must be 'none', 'mean', or 'sum'.")

if __name__ == "__main__":
    input_flow = np.random.rand(4, 4, 2)  # Example input flow map
    target_flow = np.random.rand(4, 4, 2)  # Example target flow map
    
    aepe_value = aepe(input_flow, target_flow, reduction='mean')
    print("Average Endpoint Error (AEPE):", aepe_value)