import numpy as np
from collections import Counter

def polarity(X, Y, kernel, assume_normalized_kernel=False, rescale_class_labels=False, normalize=False):
    if rescale_class_labels:
        counter = Counter(Y)
        n_pos = counter[1]
        n_neg = counter[-1]
        rescale_factor = {1: 1.0 / n_pos, -1: 1.0 / n_neg}
        Y = [rescale_factor[y] * y for y in Y]
    
    polarity_sum = 0.0
    for i in range(len(X)):
        for j in range(len(X)):
            polarity_sum += Y[i] * Y[j] * kernel(X[i], X[j])
    
    if normalize and not assume_normalized_kernel:
        polarity_sum /= len(X) * len(X)

    return polarity_sum

if __name__ == "__main__":
    def linear_kernel(x, y):
        return np.dot(x, y)
    
    X = [[1, 2], [3, 4], [5, 6]]
    Y = [1, -1, 1]
    print(polarity(X, Y, linear_kernel))
    print(polarity(X, Y, linear_kernel, rescale_class_labels=True))
    print(polarity(X, Y, linear_kernel, normalize=True))