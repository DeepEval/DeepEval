import numpy as np

def softmax(x, temperature=1.0, axis=None, shift=False):
    """
    Apply the softmax function to an array.

    Parameters:
    ----------
    x : np.ndarray
        Input array.
    temperature : float
        Temperature of the softmax function.
    axis : Optional[int]
        Axis to apply the softmax function. If None, the softmax function is
        applied to all elements of the input array.
    shift : bool
        Whether to shift the input array before applying the softmax function.
        This is useful to avoid numerical issues when the input array contains
        large values, that could result in overflows when applying the
        exponential function.

    Returns:
    -------
    np.ndarray
        Softmax function applied to the input array.
    """
    if shift:
        # Shift the input array to avoid numerical issues
        x = x - x.max(axis=axis, keepdims=True)

    # Apply the exponential function to each element
    y = np.exp(temperature * x)

    # Normalize the output array
    y = y / y.sum(axis=axis, keepdims=True)

    return y

if __name__ == "__main__":
    # Example usage
    x = np.array([1, 2, 3, 4, 5])
    y = softmax(x, temperature=2.0, axis=0, shift=True)
    print(y)