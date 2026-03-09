import numpy as np

def aepe(input, target, reduction='mean'):
    """Calculates the average endpoint error (AEPE) between two flow maps.

    Args:
        input: The input flow map with shape (*, 2).
        target: The target flow map with shape (*, 2).
        reduction: Specifies the reduction to apply to the output:
            'none': No reduction will be applied,
            'mean': The sum of the output will be divided by the number of elements in the output,
            'sum': The output will be summed.

    Returns:
        The computed AEPE as a scalar.
    """

    diff = input - target
    aepe = np.sqrt(np.sum(diff ** 2, axis=-1))

    if reduction == 'mean':
        return np.mean(aepe)
    elif reduction == 'sum':
        return np.sum(aepe)
    else:
        return aepe

if __name__ == "__main__":
    # Create sample input values
    input_flow_map = np.random.rand(100, 100, 2)
    target_flow_map = np.random.rand(100, 100, 2)

    # Calculate AEPE with reduction 'mean'
    aepe_mean = aepe(input_flow_map, target_flow_map, reduction='mean')
    print(f"AEPE (mean): {aepe_mean}")

    # Calculate AEPE with reduction 'sum'
    aepe_sum = aepe(input_flow_map, target_flow_map, reduction='sum')
    print(f"AEPE (sum): {aepe_sum}")

    # Calculate AEPE with reduction 'none'
    aepe_none = aepe(input_flow_map, target_flow_map, reduction='none')
    print(f"AEPE (none): {aepe_none}")