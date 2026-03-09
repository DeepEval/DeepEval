import numpy as np
from scipy.sparse import csr_matrix
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform

def make_multilabel_classification(n_samples, n_features, n_classes, n_labels, document_length, p_c=None, p_w_c=None):
    # Initialize arrays to store the generated samples and labels
    X = np.zeros((n_samples, n_features), dtype=np.float64)
    Y = np.zeros((n_samples, n_classes), dtype=np.bool_)

    if p_c is None:
        # Generate random class probabilities if not provided
        p_c = np.random.rand(n_classes)
        p_c /= p_c.sum()

    if p_w_c is None:
        # Generate random feature-class conditional probabilities if not provided
        p_w_c = np.random.rand(n_classes, n_features)
        p_w_c /= p_w_c.sum(axis=1)[:, np.newaxis]

    for i in range(n_samples):
        # Initialize the label set for the current sample
        label_set = np.zeros((n_classes,), dtype=np.bool_)

        # Generate the document length
        doc_len = np.random.binomial(document_length, 0.5)

        # Generate the feature values for the current sample
        for j in range(n_features):
            # Generate a random feature value for each feature in the document
            X[i, j] = np.random.choice([0, 1], size=doc_len, p=[0.5, 0.5])

        # Generate the label set for the current sample
        while True:
            # Generate a random number of labels for the current sample
            n_l = np.random.binomial(n_labels, 0.5)

            # Generate the class indices for the current sample
            class_idx = np.random.choice(n_classes, size=n_l, replace=False, p=p_c)

            # Generate the feature indices for the current sample
            feature_idx = np.random.choice(n_features, size=n_l, replace=False, p=p_w_c[class_idx, :].mean(axis=0))

            # Check if the generated label set is valid
            if len(feature_idx) == len(np.unique(feature_idx)) and len(class_idx) == len(np.unique(class_idx)):
                label_set[class_idx] = True
                break

        # Store the generated label set and feature values
        Y[i, :] = label_set
        X[i, :] = X[i, :] * label_set[:, np.newaxis]

    # Convert the label set to a sparse matrix if n_labels > 1
    if n_labels > 1:
        Y = csr_matrix(Y)

    return X, Y, p_c, p_w_c

if __name__ == "__main__":
    # Create sample input values
    n_samples = 100
    n_features = 100
    n_classes = 10
    n_labels = 3
    document_length = 10
    p_c = None
    p_w_c = None

    # Call the function and print the results
    X, Y, p_c, p_w_c = make_multilabel_classification(n_samples, n_features, n_classes, n_labels, document_length, p_c, p_w_c)
    print("Generated Samples (X): ", X)
    print("Generated Label Sets (Y): ", Y.toarray())
    print("Prior Class Probabilities (p_c): ", p_c)
    print("Conditional Probabilities of Features Given Classes (p_w_c): ", p_w_c)