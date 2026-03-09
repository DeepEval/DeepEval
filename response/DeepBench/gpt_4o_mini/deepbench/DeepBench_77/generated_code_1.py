import numpy as np

def exponential_moving_average(s, alpha=0.1, axis=0, **kwargs):
    s_ema = np.zeros_like(s)
    s_ema[0] = s[0]  # Starting point for EMA
    for i in range(1, s.shape[axis]):
        s_ema[i] = alpha * s[i] + (1 - alpha) * s_ema[i - 1]
    return s_ema

if __name__ == "__main__":
    scores = np.array([1, 2, 3, 4, 5])
    alpha = 0.2
    ema_scores = exponential_moving_average(scores, alpha=alpha)
    print("Scores:", scores)
    print("Exponential Moving Average:", ema_scores)