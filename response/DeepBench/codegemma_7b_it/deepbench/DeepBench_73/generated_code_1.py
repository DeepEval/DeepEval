from sklearn.datasets import make_multilabel_classification
import numpy as np

def make_multilabel_classification(n_samples, n_features, n_classes, n_labels, doc_length, **kwargs):
    """
    Generates a random multilabel classification problem.

    Parameters
    ----------
    n_samples : int
        The number of samples to generate.
    n_features : int
        The number of features.
    n_classes : int
        The number of classes.
    n_labels : int
        The average number of labels per sample.
    doc_length : int
        The length of the document.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        Generated samples.
    Y : ndarray or sparse matrix of shape (n_samples, n_classes)
        Label sets.
    p_c : ndarray of shape (n_classes,) (optional)
        Prior class probability.
    p_w_c : ndarray of shape (n_classes, n_features) (optional)
        Conditional probabilities of features given classes.
    """

    # Generate random samples and labels using rejection sampling
    while True:
        X, Y = make_multilabel_classification(
            n_samples, n_features, n_classes, n_labels, doc_length, **kwargs
        )
        if np.all((X >= 0) & (X <= 1)):
            break

    # Calculate prior class probability and conditional probabilities
    p_c = np.unique(Y, axis=0).shape[0] / n_samples
    p_w_c = np.zeros((n_classes, n_features))
    for i in range(n_classes):
        p_w_c[i, :] = np.mean(X[Y[:, i] == 1, :], axis=0)

    return X, Y, p_c, p_w_c

if __name__ == "__main__":
    # Example usage
    n_samples = 1000
    n_features = 100
    n_classes = 10
    n_labels = 3
    doc_length = 200

    X, Y, p_c, p_w_c = make_multilabel_classification(
        n_samples, n_features, n_classes, n_labels, doc_length
    )

    print("X:", X.shape)
    print("Y:", Y.shape)
    print("p_c:", p_c.shape)
    print("p_w_c:", p_w_c.shape)