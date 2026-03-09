import numpy as np
from scipy.stats import norm
from sklearn.datasets import make_classification

def make_hastie_10_2(n_samples=100):
    # generate random x1 and x2 values from normal distributions
    x1 = np.random.randn(n_samples) + 0.5
    x2 = np.random.randn(n_samples) + 0.5
    
    # generate y based on the Hastie et al. 2009 dataset
    y = np.where((x1 - 1)**2 + x2**2 > 0.6, 0, 1)
    
    # create a classification dataset with the generated features and labels
    X = np.column_stack((x1, x2))
    return X, y

if __name__ == "__main__":
    n_samples = 100
    X, y = make_hastie_10_2(n_samples)
    print("Features (X):\n", X)
    print("Labels (y):\n", y)
    
    # Create a classification dataset using Scikit-learn for comparison
    X_sklearn, y_sklearn = make_classification(n_samples=n_samples)
    print("\nFeatures (X) from Scikit-learn:\n", X_sklearn)
    print("Labels (y) from Scikit-learn:\n", y_sklearn)