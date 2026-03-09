import numpy as np

def softmax_func(x, temperature=1.0, axis=None, shift=False):
    """
    Softmax function applied to the input array.

    Parameters:
    x (np.ndarray): Input array.
    temperature (float): Temperature of the softmax function.
    axis (Optional[int]): Axis to apply the softmax function. If None, the function is applied to all elements of the array.
    shift (bool): Whether to shift the input array before applying the softmax function.

    Returns:
    np.ndarray: Softmax function applied to the input array.
    """
    x = x - np.max(x, axis=axis, keepdims=True) if shift else x
    exp_x = np.exp(x / temperature)
    softmax_values = exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    return softmax_values

if __name__ == "__main__":
    # Create sample input values
    x = np.array([1.0, 2.0, 3.0])
    temperature = 0.5
    axis = None
    shift = False

    # Call the function and print the results
    result = softmax_func(x, temperature, axis, shift)
    print("Softmax applied to x:", result)