import numpy as np

def exponential_moving_average(s, alpha=0.1, axis=0, **kwargs):
    s = np.asarray(s)
    axis = axis % s.ndim
    s_ema = np.zeros_like(s)
    s_ema[:, 0] = s[:, 0]
    for i in range(1, s.shape[axis]):
        idx = [slice(None)] * s.ndim
        idx[axis] = i
        s_ema[idx] = alpha * s[idx] + (1 - alpha) * s_ema[idx[:-1]]
    return s_ema

if __name__ == "__main__":
    # Generate sample input values
    s = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])

    # Calculate the exponential moving average
    alpha = 0.1
    s_ema = exponential_moving_average(s, alpha=alpha)

    # Print the results
    print("Input scores:")
    print(s)
    print("Exponential moving average scores:")
    print(s_ema)