import numpy as np
from scipy.sparse import lil_matrix

def make_multilabel_classification(n_samples, n_features, n_classes, n_labels, doc_length, prior_class_prob=None, cond_prob_wc=None):
    # Generate a random multilabel classification problem

    # Initialize an ndarray of zeros for the sample data
    X = np.zeros((n_samples, n_features))

    # Initialize a list to store the label sets for each sample
    Y = []

    # Loop through each sample
    for i in range(n_samples):
        # Generate a random label for each sample
        label = np.random.choice(n_classes, size=n_labels, replace=True)

        # Append the label to the list of labels for this sample
        Y.append(label)

        # Add the label to the sample data
        X[i, label] = 1

    # Convert the list of labels to an ndarray
    Y = np.asarray(Y)

    # Check if prior class probabilities are provided
    if prior_class_prob is not None:
        # Generate random values for the prior class probabilities
        prior_class_prob = np.random.uniform(size=n_classes)

    # Check if conditional probabilities of features given classes are provided
    if cond_prob_wc is not None:
        # Generate random values for the conditional probabilities
        cond_prob_wc = np.random.uniform(size=(n_classes, n_features))

    # Return the generated samples and labels
    return X, Y, prior_class_prob, cond_prob_wc

if __name__ == "__main__":
    # Generate sample input values
    n_samples = 100
    n_features = 10
    n_classes = 5
    n_labels = 3
    doc_length = 10
    prior_class_prob = None
    cond_prob_wc = None

    # Call the function and print the results
    X, Y, prior_class_prob, cond_prob_wc = make_multilabel_classification(n_samples, n_features, n_classes, n_labels, doc_length, prior_class_prob, cond_prob_wc)
    print(X.shape)
    print(Y.shape)
    print(prior_class_prob)
    print(cond_prob_wc)