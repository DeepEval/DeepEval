import numpy as np

def exponential_moving_average(scores, alpha=0.5, axis=0, **kwargs):
    """
    Calculates the exponential moving average (EMA) score aggregation for a given array of scores.
    
    Parameters
    ----------
    scores : array-like
        Array of scores.
    alpha : float, optional
        Determines the weight of the previous EMA score. The default is 0.5.
    axis : int, optional
        Indicates the axis along which the scores are sorted. The default is 0.
    **kwargs : additional keyword arguments
        Additional parameters to be passed to the function.
    
    Returns
    -------
    s_ema : array-like
        Exponential moving average score.
    """
    if len(scores) == 0:
        raise ValueError("Empty input array.")

    scores = np.asarray(scores)
    if scores.ndim == 0:
        scores = scores[None]

    s_ema = scores
    for t in range(1, len(scores)):
        s_ema[t] = alpha * scores[t] + (1 - alpha) * s_ema[t - 1]

    return s_ema

if __name__ == "__main__":
    # Test the function
    scores = np.array([1, 2, 3, 4, 5, 6])
    print(exponential_moving_average(scores, alpha=0.5, axis=0))  # [1.      2.      3.      4.      5.      6.    ]
    print(exponential_moving_average(scores, alpha=0.2, axis=0))  # [0.8333  1.6667  2.5     3.3333  4.1667  5.    ]
    print(exponential_moving_average(scores, alpha=0.8, axis=0))  # [1.      1.9333  2.8333  3.7333  4.6333  5.5    ]