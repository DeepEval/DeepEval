import numpy as np
from scipy.sparse import random as sparse_random

def make_multilabel_classification(n_samples=100, n_features=20, n_classes=5, n_labels=2, document_length=10, density=0.1):
    X = np.random.rand(n_samples, n_features)
    
    if n_labels > n_classes:
        n_labels = n_classes

    Y = np.zeros((n_samples, n_classes))
    
    for i in range(n_samples):
        labels = np.random.choice(n_classes, n_labels, replace=False)
        Y[i, labels] = 1

    # Optionally, create prior class probabilities
    p_c = np.mean(Y, axis=0)
    
    # Optionally, create conditional probabilities of features given classes
    p_w_c = np.zeros((n_features, n_classes))
    
    for c in range(n_classes):
        if np.sum(Y[:, c]) > 0:
            p_w_c[:, c] = np.random.rand(n_features) * (Y[:, c] @ X / np.sum(Y[:, c]))  # simulate conditional probabilities

    return X, Y, p_c, p_w_c

if __name__ == "__main__":
    X, Y, p_c, p_w_c = make_multilabel_classification(n_samples=10, n_features=5, n_classes=3, n_labels=2)
    print("Features (X):")
    print(X)
    print("\nLabels (Y):")
    print(Y)
    print("\nPrior Class Probabilities (p_c):")
    print(p_c)
    print("\nConditional Probabilities (p_w_c):")
    print(p_w_c)