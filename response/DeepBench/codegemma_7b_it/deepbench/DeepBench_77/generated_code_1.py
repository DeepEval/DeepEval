import numpy as np

def exponential_moving_average(s, alpha=0.9, axis=0, **kwargs):
    s_ema = np.copy(s)
    for i in range(1, s.shape[axis]):
        mask = np.arange(i) + 1
        s_ema[mask, ...] = alpha * s[mask, ...] + (1 - alpha) * s_ema[mask - 1, ...]
    return s_ema

if __name__ == "__main__":
    # Example usage:
    s = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    s_ema = exponential_moving_average(s)
    print(s_ema)