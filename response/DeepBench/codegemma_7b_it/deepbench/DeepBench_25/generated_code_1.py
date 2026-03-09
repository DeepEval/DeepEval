import numpy as np
from sklearn.utils import check_array, check_random_state

def aom(scores, n_buckets=5, method='static', bootstrap_estimators=False, random_state=None):
    random_state = check_random_state(random_state)
    scores = check_array(scores, accept_sparse=False, ensure_2d=True)
    n_samples, n_estimators = scores.shape

    if method == 'static' and n_estimators % n_buckets != 0:
        raise ValueError("Number of estimators must be divisible by the number of buckets.")

    if method == 'dynamic' and n_estimators < n_buckets:
        raise ValueError("Number of estimators must be greater than or equal to the number of buckets.")

    if method not in ('static', 'dynamic'):
        raise ValueError("Invalid method: must be 'static' or 'dynamic'.")

    if bootstrap_estimators and method == 'static':
        raise ValueError("Bootstrap estimators not supported for static method.")

    estimators_per_bucket = n_estimators // n_buckets
    if method == 'static':
        estimator_indices = np.arange(n_estimators).reshape((n_buckets, estimators_per_bucket))
    else:
        if bootstrap_estimators:
            estimator_indices = random_state.randint(n_estimators, size=(n_buckets, estimators_per_bucket))
        else:
            estimator_indices = np.arange(n_estimators).reshape((n_buckets, estimators_per_bucket))

    max_scores = np.zeros(n_samples)
    for bucket_idx in range(n_buckets):
        bucket_scores = scores[:, estimator_indices[bucket_idx]]
        max_scores += np.max(bucket_scores, axis=1)

    combined_scores = max_scores / n_buckets
    return combined_scores

if __name__ == "__main__":
    # Sample input values
    scores = np.array([[0.5, 0.3, 0.7],
                        [0.2, 0.4, 0.9],
                        [0.6, 0.8, 0.4]])

    # Call the function
    combined_scores = aom(scores, n_buckets=2)

    # Print the results
    print(combined_scores)