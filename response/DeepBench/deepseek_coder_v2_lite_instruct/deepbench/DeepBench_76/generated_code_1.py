import numpy as np

def combo_moa(scores, n_buckets, method, bootstrap_estimators, random_state):
    n_samples, n_estimators = scores.shape
    if method == 'static':
        indices = np.argsort(scores, axis=1)
    elif method == 'dynamic':
        if bootstrap_estimators:
            random_state = np.random.RandomState(random_state)
            indices = random_state.rand(*scores.shape).argsort(axis=1)
        else:
            indices = np.argsort(scores, axis=1)
    else:
        raise ValueError("Invalid method specified. Choose 'static' or 'dynamic'.")

    bucket_indices = np.linspace(0, n_estimators, n_buckets + 1).astype(int)
    combined_scores = np.zeros((n_samples,))

    for i in range(n_samples):
        for j in range(n_buckets):
            combined_scores[i] += np.mean(scores[i, indices[i, bucket_indices[j]:bucket_indices[j+1]]])

    return combined_scores

def moa(scores, n_buckets=5, method='static', bootstrap_estimators=False, random_state=None):
    return combo_moa(scores, n_buckets, method, bootstrap_estimators, random_state)

if __name__ == "__main__":
    scores = np.array([[0.1, 0.4, 0.3, 0.2, 0.5],
                       [0.3, 0.2, 0.4, 0.1, 0.5],
                       [0.2, 0.3, 0.4, 0.1, 0.5]])
    combined_scores = moa(scores)
    print(combined_scores)