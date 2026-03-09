import numpy as np

def exponential_moving_average(s, alpha=0.1, axis=0, **kwargs):
    s = np.array(s)
    if axis < 0:
        axis += s.ndim
    
    s_ema = np.take(s, 0, axis=axis).astype(float)
    s_shape = list(s.shape)
    del s_shape[axis]
    
    for i in np.ndindex(*s_shape):
        index = list(i)
        index.insert(axis, slice(None))
        sub_array = s[tuple(index)]
        
        ema = sub_array[0]
        ema_array = np.zeros_like(sub_array, dtype=float)
        ema_array[0] = ema
        
        for t in range(1, len(sub_array)):
            ema = alpha * sub_array[t] + (1 - alpha) * ema
            ema_array[t] = ema
        
        s_ema[tuple(i)] = ema_array

    return s_ema

if __name__ == "__main__":
    scores = [1, 2, 3, 4, 5]
    ema_result = exponential_moving_average(scores, alpha=0.2)
    print("Exponential Moving Average:", ema_result)

    scores_2d = np.array([[1, 2, 3], [4, 5, 6]])
    ema_result_2d = exponential_moving_average(scores_2d, alpha=0.2, axis=1)
    print("Exponential Moving Average (2D):")
    print(ema_result_2d)