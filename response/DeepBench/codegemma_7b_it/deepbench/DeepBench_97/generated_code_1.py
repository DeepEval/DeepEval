import numpy as np

def softmax(x, temperature=1.0, axis=None, shift=False):
    x = np.asarray(x)

    if shift:
        x = x - np.max(x, axis=axis, keepdims=True)

    if axis is None:
        x = np.exp(x / temperature)
        return x / np.sum(x, axis=None)
    else:
        x = np.exp(x / temperature)
        return x / np.sum(x, axis=axis, keepdims=True)

if __name__ == "__main__":
    # Example usage:
    x = np.array([1, 2, 3, 4])
    y = softmax(x, temperature=0.5)
    print(y)