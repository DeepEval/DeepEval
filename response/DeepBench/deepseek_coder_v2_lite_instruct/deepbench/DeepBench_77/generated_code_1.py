import numpy as np

def exponential_moving_average(s, alpha=0.5, axis=0, **kwargs):
    s_ema = np.zeros_like(s, dtype=float)
    s_ema[0] = s[0]
    for t in range(1, s.shape[axis]):
        s_ema[t] = alpha * s[t] + (1 - alpha) * s_ema[t - 1]
    return s_ema

if __name__ == "__main__":
    scores = np.array([10, 20, 30, 40, 50])
    ema_scores = exponential_moving_average(scores, alpha=0.7)
    print("Exponential Moving Average Scores:", ema_scores)