import numpy as np
from sklearn.utils import resample

def combo_moa(scores, n_buckets, method, bootstrap_estimators, random_state):
    rng = np.random.RandomState(random_state)
    n_samples, n_estimators = scores.shape
    combined_scores = np.zeros(n_samples)

    if method == 'static':
        for i in range(n_buckets):
            if bootstrap_estimators:
                selected_indices = rng.choice(n_estimators, size=n_estimators, replace=True)
            else:
                selected_indices = rng.choice(n_estimators, size=n_estimators, replace=False)

            bucket_scores = scores[:, selected_indices].mean(axis=1)
            combined_scores += bucket_scores
    elif method == 'dynamic':
        bucket_size = n_estimators // n_buckets
        for i in range(n_buckets):
            if i == n_buckets - 1:
                selected_indices = np.arange(i * bucket_size, n_estimators)
            else:
                selected_indices = np.arange(i * bucket_size, (i + 1) * bucket_size)

            if bootstrap_estimators:
                selected_indices = rng.choice(selected_indices, size=len(selected_indices), replace=True)
            
            bucket_scores = scores[:, selected_indices].mean(axis=1)
            combined_scores += bucket_scores

    combined_scores /= n_buckets
    return combined_scores

def moa(scores, n_buckets=5, method='static', bootstrap_estimators=False, random_state=None):
    return combo_moa(scores, n_buckets, method, bootstrap_estimators, random_state)

if __name__ == "__main__":
    np.random.seed(42)
    scores = np.random.rand(10, 4)  # Example with 10 samples and 4 estimators
    combined_scores = moa(scores, n_buckets=3, method='static', bootstrap_estimators=True, random_state=42)
    print("Combined Scores:", combined_scores)