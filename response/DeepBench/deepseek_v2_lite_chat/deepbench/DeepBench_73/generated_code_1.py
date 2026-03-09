import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.datasets import make_multilabel
import matplotlib.pyplot as plt

def make_multilabel_classification(n_samples=100, n_features=50, n_classes=5, n_labels=3, document_length=10, p_c=None, p_w_c=None, random_state=None):
    """
    Generate a random multilabel classification problem.

    Args:
    n_samples (int): Number of samples (default is 100).
    n_features (int): Number of features per sample (default is 50).
    n_classes (int): Number of classes (default is 5).
    n_labels (int): Number of labels per sample (default is 3).
    document_length: Length of the document (default is 10).
    p_c (dict or None): Prior class probabilities (default is None).
    p_w_c (dict or None): Conditional probabilities of features given classes (default is None).
    random_state (int or None): Seed for the random number generator (default is None).

    Returns:
    tuple: A tuple containing the generated samples, labels, and optional prior class probabilities and conditional probabilities.
    """
    # Seed for reproducibility
    np.random.seed(random_state)
    
    # Create a random multilabel classification problem
    X, Y, p_c, p_w_c = make_multilabel(n_samples=n_samples, n_features=n_features, n_classes=n_classes, n_labels=n_labels, random_state=random_state)
    
    # Convert to numpy arrays
    X = X.astype(np.float32)
    Y = Y.astype(np.int32)
    
    # Normalize features if necessary
    if p_w_c is not None:
        for key in p_w_c.keys():
            X[:, key] = p_w_c[key] * X[:, key]
    
    # Apply prior probabilities if provided
    if p_c is not None:
        Y = np.random.choice(np.arange(n_classes), size=Y.shape, p=p_c)
    
    return X, Y, p_c, p_w_c

if __name__ == "__main__":
    # Sample input values
    n_samples = 100
    n_features = 50
    n_classes = 5
    n_labels = 3
    document_length = 10
    prior_prob = {0: 0.2, 1: 0.3, 2: 0.5}
    conditional_prob = {(0, 0): 0.5, (0, 1): 0.5, (1, 0): 0.5, (1, 1): 0.5, (2, 0): 0.5, (2, 1): 0.5, (2, 2): 0.5}

    # Call the function and print the results
    X, Y, p_c, p_w_c = make_multilabel_classification(n_samples=n_samples, n_features=n_features, n_classes=n_classes, n_labels=n_labels, document_length=document_length, p_c=prior_prob, p_w_c=conditional_prob)
    
    # Plot the first few samples
    plt.figure(figsize=(10, 10))
    for i in range(10):
        plt.subplot(5, 5, i + 1)
        plt.imshow(X[i].reshape(document_length, document_length), cmap='gray_r')
        plt.xticks([])
        plt.yticks([])
        plt.xlabel(np.argmax(Y[i]), fontsize=10)
    plt.tight_layout()
    plt.show()