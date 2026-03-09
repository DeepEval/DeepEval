import numpy as np

def softmax(x: np.ndarray, temperature: float = 1.0, axis: int = None, shift: bool = True) -> np.ndarray:
    if shift:
        x_max = np.max(x, axis=axis, keepdims=True)
        x = x - x_max

    exp_x = np.exp(x / temperature)
    sum_exp_x = np.sum(exp_x, axis=axis, keepdims=True)
    return exp_x / sum_exp_x

if __name__ == "__main__":
    # Example input
    input_array = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    temperature = 1.0
    axis = 1

    # Call softmax function
    result = softmax(input_array, temperature=temperature, axis=axis, shift=True)

    # Print the result
    print("Softmax result:")
    print(result)