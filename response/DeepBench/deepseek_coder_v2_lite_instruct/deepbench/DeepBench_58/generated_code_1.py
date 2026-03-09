import numpy as np

def polarity(X, Y, kernel, assume_normalized_kernel=False, rescale_class_labels=False, normalize=False):
    if rescale_class_labels:
        unique, counts = np.unique(Y, return_counts=True)
        rescaling_factor = len(Y) / (2 * max(counts))
        Y = np.where(Y == unique[0], -1, 1)
    elif not assume_normalized_kernel:
        Y = np.where(Y == -1, -1, 1)

    K = np.array([[kernel(x, y) for y in X] for x in X])
    
    if normalize:
        K = (K - np.mean(K)) / np.std(K)

    return np.sum(Y[:, np.newaxis] * K * Y[np.newaxis, :])

if __name__ == "__main__":
    # Example usage
    X = [1, 2, 3, 4]
    Y = [-1, -1, 1, 1]
    
    def kernel(x, y):
        return x * y
    
    print(polarity(X, Y, kernel))  # Expected output: 4 (sum of products of class labels and kernel values)