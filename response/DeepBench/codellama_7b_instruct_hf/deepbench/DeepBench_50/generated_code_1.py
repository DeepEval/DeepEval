import numpy as np

def aepe(input, target, reduction='none'):
    """
    Calculates the average endpoint error (AEPE) between two flow maps.

    Args:
        input (np.ndarray): the input flow map with shape (*, 2)
        target (np.ndarray): the target flow map with shape (*, 2)
        reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'
            'none': no reduction will be applied, 'mean': the sum of the output will be divided by the number of elements in the output, 'sum': the output will be summed.

    Returns:
        the computed AEPE as a scalar
    """
    # Calculate the difference between the two flow maps
    diff = input - target

    # Calculate the squared difference between the two flow maps
    squared_diff = diff ** 2

    # Calculate the sum of the squared differences
    sum_squared_diff = np.sum(squared_diff, axis=(1, 2))

    # Calculate the average of the squared differences
    ave_squared_diff = sum_squared_diff / (input.shape[1] * input.shape[2])

    # Calculate the average endpoint error
    aepe = np.sqrt(ave_squared_diff)

    # Apply the reduction
    if reduction == 'mean':
        aepe = np.mean(aepe)
    elif reduction == 'sum':
        aepe = np.sum(aepe)

    return aepe

if __name__ == "__main__":
    # Create sample input values
    input = np.array([
        [
            [1, 2],
            [3, 4],
            [5, 6]
        ],
        [
            [7, 8],
            [9, 10],
            [11, 12]
        ]
    ])

    target = np.array([
        [
            [1, 2],
            [3, 4],
            [5, 6]
        ],
        [
            [7, 8],
            [9, 10],
            [11, 12]
        ]
    ])

    # Call the function
    aepe = aepe(input, target)

    # Print the results
    print(aepe)