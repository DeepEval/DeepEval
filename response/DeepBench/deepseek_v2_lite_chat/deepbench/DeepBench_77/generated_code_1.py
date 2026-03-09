import numpy as np
import pandas as pd

def exponential_moving_average(s, alpha=0.5, axis=0, **kwargs):
    """
    Calculates the exponential moving average of a given array of scores.
    
    Parameters:
        s (array): The array of scores.
        alpha (float): The forgetting factor (default is 0.5).
        axis (int): The axis along which to compute the EMA (default is 0).
        **kwargs: Additional keyword arguments.
    
    Returns:
        array: The exponential moving average score.
    """
    s = np.array(s, copy=True)
    if s.ndim == 1:
        s = s[:, np.newaxis]
    
    # Ensure the axis is valid
    if axis not in [0, 1]:
        raise ValueError("Invalid axis value. It should be 0 or 1.")
    
    # Compute EMA using the formula
    ema = np.where(np.abs(s - s[:, ::-1]) < 1e-08, alpha * s + (1 - alpha) * np.average(s, axis=axis), ema)
    
    return ema

if __name__ == "__main__":
    # Sample input array
    scores = np.array([10, 12, 15, 14, 15, 17, 20, 22, 25, 27])
    
    # Calculate EMA using the function
    ema_scores = exponential_moving_average(scores, alpha=0.7, axis=1)
    
    # Print the results
    print("Original Scores:", scores)
    print("Exponential Moving Average Scores:", ema_scores)