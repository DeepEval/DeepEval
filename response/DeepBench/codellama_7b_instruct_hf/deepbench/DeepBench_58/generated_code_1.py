import numpy as np

def polarity(X, Y, kernel, assume_normalized_kernel=False, rescale_class_labels=False, normalize=False):
    if assume_normalized_kernel:
        kernel = kernel / np.max(np.abs(kernel))
    if rescale_class_labels:
        Y = np.array(Y)
        Y[Y == 1] = 0.5
        Y[Y == -1] = -0.5
    if normalize:
        X = X / np.max(np.abs(X))
    kernel_polarity = np.sum(Y * kernel(X, X))
    return kernel_polarity

if __name__ == "__main__":
    X = np.array([[1, 2], [3, 4], [5, 6]])
    Y = np.array([1, -1, 1])
    kernel = lambda x, y: np.sum(x * y)
    kernel_polarity = polarity(X, Y, kernel, normalize=True)
    print(kernel_polarity)