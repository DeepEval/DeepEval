import numpy as np

def aepe(input_flow_map, target_flow_map, reduction='none'):
    """
    Calculate the average endpoint error (AEPE) between two flow maps.
    
    Args:
        input_flow_map (np.ndarray): The input flow map with shape (h, w, 2).
        target_flow_map (np.ndarray): The target flow map with shape (h, w, 2).
        reduction (str): Specifies the reduction to apply to the output.
            - 'none': no reduction will be applied.
            - 'mean': the sum of the output will be divided by the number of elements in the output.
            - 'sum': the output will be summed.
    
    Returns:
        np.float64: The computed AEPE as a scalar.
    """
    # Ensure both input and target flow maps have the same shape
    assert input_flow_map.shape == target_flow_map.shape, "Input and target flow maps must have the same shape"
    
    # Calculate the difference between the input and target flow maps for each component
    diff = (input_flow_map - target_flow_map) ** 2
    
    # Sum over the channels (assuming color images with 3 channels)
    channel_sum = np.sum(diff, axis=(0, 1))  # axis=(0, 1) sums over the channels
    
    # Calculate the average endpoint error (AEPE)
    aepe_value = np.sqrt(np.sum(channel_sum) / (input_flow_map.shape[0] * input_flow_map.shape[1]))
    
    # Apply reduction
    if reduction == 'mean':
        aepe_value = np.mean(aepe_value)
    elif reduction == 'sum':
        aepe_value = np.sum(aepe_value)
    
    return aepe_value

if __name__ == "__main__":
    import numpy as np
    
    # Sample input values
    input_flow_map = np.random.rand(5, 5, 2)  # Random optical flow for a 5x5 image
    target_flow_map = np.random.rand(5, 5, 2)  # Random target optical flow
    
    # Calculate AEPE
    aepe_value = aepe(input_flow_map, target_flow_map, reduction='mean')
    
    # Print the results
    print("AEPE:", aepe_value)