import numpy as np

def softmax(x, temperature, axis=None, shift=True):
    if shift:
        x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x / temperature)
    if axis is not None:
        exp_x = exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    else:
        exp_x = exp_x / np.sum(exp_x)
    return exp_x

if __name__ == "__main__":
    # Create sample input values
    x = np.array([[1, 2, 3], [4, 5, 6]])
    
    # Call the function
    result = softmax(x, temperature=1.0, axis=1)
    
    # Print the results
    print(result)
    
    # Test with shifted input
    result_shifted = softmax(x, temperature=1.0, axis=1, shift=True)
    
    # Print the results
    print(result_shifted)