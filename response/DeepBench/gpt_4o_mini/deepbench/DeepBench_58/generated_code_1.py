import numpy as np

def polarity(X, Y, kernel, assume_normalized_kernel=False, rescale_class_labels=False, normalize=False):
    if rescale_class_labels:
        count_pos = np.sum(np.array(Y) == 1)
        count_neg = np.sum(np.array(Y) == -1)
        total_count = len(Y)
        Y_rescaled = np.array([1 / count_pos if label == 1 else -1 / count_neg for label in Y])
    else:
        Y_rescaled = np.array(Y)

    kernel_values = np.array([[kernel(x_i, x_j) for x_j in X] for x_i in X])
    polarity_value = np.sum(Y_rescaled[:, None] * Y_rescaled[None, :] * kernel_values)

    if normalize:
        polarity_value /= total_count
    
    return polarity_value

if __name__ == "__main__":
    def sample_kernel(x1, x2):
        return np.dot(x1, x2)

    X = [np.array([1, 2]), np.array([2, 3]), np.array([3, 4])]
    Y = [1, -1, 1]
    
    result = polarity(X, Y, sample_kernel, rescale_class_labels=True, normalize=True)
    print("Kernel Polarity:", result)