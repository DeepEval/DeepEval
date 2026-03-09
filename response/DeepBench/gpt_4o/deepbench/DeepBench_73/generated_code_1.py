import numpy as np
from scipy.sparse import csr_matrix

def make_multilabel_classification(n_samples=100, n_features=20, n_classes=5, n_labels=2, 
                                   length=50, allow_unlabeled=False, sparse_output=False,
                                   return_probabilities=False, random_state=None):
    rng = np.random.default_rng(random_state)

    if return_probabilities:
        p_c = rng.random(n_classes)
        p_c /= p_c.sum()
        p_w_c = rng.random((n_features, n_classes))
        p_w_c /= p_w_c.sum(axis=0)
    else:
        p_c = None
        p_w_c = None

    X = np.zeros((n_samples, n_features))
    Y = np.zeros((n_samples, n_classes))

    for i in range(n_samples):
        while True:
            labels = rng.choice(n_classes, size=n_labels, replace=False)
            if not allow_unlabeled or len(labels) > 0:
                break

        Y[i, labels] = 1

        for label in labels:
            words = rng.choice(n_features, size=length // n_labels, replace=False)
            X[i, words] += 1

    if sparse_output:
        X = csr_matrix(X)
        Y = csr_matrix(Y)

    return (X, Y) if not return_probabilities else (X, Y, p_c, p_w_c)

if __name__ == "__main__":
    n_samples = 10
    n_features = 15
    n_classes = 3
    n_labels = 2
    length = 10

    X, Y, p_c, p_w_c = make_multilabel_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        n_labels=n_labels,
        length=length,
        return_probabilities=True,
        random_state=42
    )

    print("Generated feature matrix X:")
    print(X)
    print("\nGenerated label matrix Y:")
    print(Y)
    print("\nPrior class probabilities p_c:")
    print(p_c)
    print("\nConditional probabilities p_w_c:")
    print(p_w_c)