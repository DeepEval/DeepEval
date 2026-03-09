import numpy as np

def softmax(x, temperature=1.0, axis=None, shift=False):
    if shift:
        x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x / temperature)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

if __name__ == "__main__":
    x = np.array([1, 2, 3])
    print("Input:", x)
    print("Softmax Output:", softmax(x))
    print("Temperature = 0.5:", softmax(x, temperature=0.5))
    print("Shifted Input:", softmax(x, shift=True))