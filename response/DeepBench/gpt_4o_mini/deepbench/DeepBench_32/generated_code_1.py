import numpy as np
import pandas as pd

def make_hastie_10_2(n_samples=1000):
    np.random.seed(0)
    x = np.random.normal(size=(n_samples, 10))
    y = np.zeros(n_samples)
    
    for i in range(n_samples):
        y[i] = (np.sum(np.square(x[i, :5])) > np.sum(np.square(x[i, 5:]))) * 2 - 1
    
    return pd.DataFrame(x, columns=[f'feature_{i}' for i in range(10)]), y

if __name__ == "__main__":
    X, y = make_hastie_10_2(10)
    print("Features:\n", X)
    print("Labels:\n", y)