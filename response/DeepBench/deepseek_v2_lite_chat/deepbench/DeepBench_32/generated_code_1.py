import numpy as np
from sklearn.datasets import make_hastie_10_2

def make_hastie_10_2(size=100, random_state=None):
    """
    Generate binary classification data using the Hastie et al. 2009 dataset.
    
    Args:
        size (int): The number of samples to generate.
        random_state (int): A seed for the random number generator.
    
    Returns:
        X (numpy array): The input data.
        y (numpy array): The corresponding labels.
    """
    X, y = make_hastie_10_2(size=size, random_state=random_state)
    return X, y

if __name__ == "__main__":
    # Create sample input values
    X, y = make_hastie_10_2(size=1000, random_state=42)
    
    # Call the function and print the results
    X_hastie, y_hastie = make_hastie_10_2(size=1000, random_state=42)
    print("Original Hastie Dataset:")
    print(f"Input Shape: {X.shape}")
    print(f"Labels Shape: {y.shape}")
    print(f"First 5 Samples:")
    print(X[:5])
    print(y[:5])