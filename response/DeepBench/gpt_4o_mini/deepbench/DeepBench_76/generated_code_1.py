import numpy as np

def combo_moa(scores, n_buckets, method, bootstrap_estimators, random_state):
    n_samples, n_estimators = scores.shape
    if method == 'static':
        indices = np.arange(n_samples)
        np.random.seed(random_state)
        bucket_indices = np.array_split(np.random.permutation(indices), n_buckets)
    else:  # dynamic
        if bootstrap_estimators:
            np.random.seed(random_state)
            bucket_indices = [np.random.choice(n_samples, size=n_samples, replace=True) for _ in range(n_buckets)]
        else:
            bucket_indices = [np.arange(n_samples) for _ in range(n_buckets)]

    combined_scores = np.zeros(n_samples)
    for bucket in bucket_indices:
        combined_scores[bucket] = np.mean(scores[bucket], axis=1)

    return combined_scores

def moa(scores, n_buckets=5, method='static', bootstrap_estimators=False, random_state=None):
    return combo_moa(scores, n_buckets, method, bootstrap_estimators, random_state)

if __name__ == "__main__":
    np.random.seed(0)
    scores = np.random.rand(10, 3)  # 10 samples, 3 estimators
    combined_scores = moa(scores, n_buckets=5, method='static', bootstrap_estimators=False, random_state=42)
    print("Combined Scores:", combined_scores)