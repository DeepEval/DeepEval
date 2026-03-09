import numpy as np

def aom(scores, n_buckets=5, method='static', bootstrap_estimators=False, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)

    n_samples, n_estimators = scores.shape
    combined_scores = np.zeros(n_samples)

    if method == 'static':
        bucket_indices = np.array_split(np.arange(n_estimators), n_buckets)
    elif method == 'dynamic':
        bucket_indices = []
        for i in range(n_buckets):
            if bootstrap_estimators:
                indices = np.random.choice(n_estimators, size=n_estimators, replace=True)
            else:
                indices = np.random.choice(n_estimators, size=n_estimators, replace=False)
            bucket_indices.append(indices)
    else:
        raise ValueError("Method must be either 'static' or 'dynamic'.")

    for indices in bucket_indices:
        max_scores = np.max(scores[:, indices], axis=1)
        combined_scores += max_scores

    combined_scores /= n_buckets
    return combined_scores

if __name__ == "__main__":
    np.random.seed(42)
    scores = np.random.rand(10, 3)  # 10 samples, 3 estimators
    combined_scores = aom(scores, n_buckets=2, method='static', bootstrap_estimators=False)
    print("Combined Scores:", combined_scores)