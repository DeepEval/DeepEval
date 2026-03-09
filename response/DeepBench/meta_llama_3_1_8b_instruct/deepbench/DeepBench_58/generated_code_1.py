import numpy as np

def polarity(X, Y, kernel, assume_normalized_kernel=False, rescale_class_labels=False, normalize=True):
    if assume_normalized_kernel:
        kernel_values = kernel(X)
    else:
        kernel_values = np.array([kernel(x, X) for x in X])

    if rescale_class_labels:
        class_counts = np.unique(Y, return_counts=True)[1]
        Y = Y * class_counts.max() / class_counts

    if normalize:
        kernel_values = kernel_values / np.max(kernel_values)

    kernel_polarity = np.sum(Y[:, np.newaxis] * Y[np.newaxis, :] * kernel_values)
    return kernel_polarity

if __name__ == "__main__":
    X = [1, 2, 3, 4, 5]
    Y = [-1, 1, -1, 1, -1]
    kernel = lambda x, X: np.sum(np.array(X) ** 2)
    print(polarity(X, Y, kernel))
    print(polarity(X, Y, kernel, rescale_class_labels=True))
    print(polarity(X, Y, kernel, normalize=False))