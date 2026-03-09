import numpy as np

def exponential_moving_average(s, alpha=0.1, axis=-1, **kwargs):
    s_ema = np.zeros_like(s, dtype=np.float32)
    s_ema[..., 0] = s[..., 0]
    for t in range(1, s.shape[axis]):
        s_ema[..., t] = alpha * s[..., t] + (1 - alpha) * s_ema[..., t - 1]
    return s_ema

if __name__ == "__main__":
    scores = np.array([10, 12, 15, 13, 18, 20])
    ema_scores = exponential_moving_average(scores, alpha=0.3)
    print(f"Scores: {scores}")
    print(f"EMA Scores: {ema_scores}")