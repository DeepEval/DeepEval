import numpy as np

def softmax(x: np.ndarray, temperature: float = 1.0, axis: int = None, shift: bool = True) -> np.ndarray:
    if shift:
        x = x - np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x / temperature)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

if __name__ == "__main__":
    sample_input = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    temperature = 1.0
    result = softmax(sample_input, temperature=temperature, axis=1, shift=True)
    print("Softmax Result:\n", result)