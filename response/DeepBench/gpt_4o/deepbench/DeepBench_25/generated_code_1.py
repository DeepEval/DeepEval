import numpy as np

def aom(scores, n_buckets=5, method='static', bootstrap_estimators=False, random_state=None):
    n_samples, n_estimators = scores.shape
    rng = np.random.default_rng(random_state)

    if method not in ['static', 'dynamic']:
        raise ValueError("method must be either 'static' or 'dynamic'")
    
    if method == 'static':
        bucket_size = n_estimators // n_buckets
        buckets = [scores[:, i * bucket_size:(i + 1) * bucket_size] for i in range(n_buckets)]
    else:  # dynamic method
        buckets = []
        for _ in range(n_buckets):
            if bootstrap_estimators:
                indices = rng.choice(n_estimators, size=n_estimators, replace=True)
            else:
                indices = rng.permutation(n_estimators)
            buckets.append(scores[:, indices[:n_estimators // n_buckets]])

    max_scores = np.array([np.max(bucket, axis=1) for bucket in buckets]).T
    combined_scores = np.mean(max_scores, axis=1)

    return combined_scores

if __name__ == "__main__":
    # Sample input values
    np.random.seed(42)
    scores = np.random.rand(10, 20)  # 10 samples, 20 estimators
    n_buckets = 4
    method = 'static'
    bootstrap_estimators = True
    random_state = 42

    # Call the function and print results
    combined_scores = aom(scores, n_buckets=n_buckets, method=method, 
                          bootstrap_estimators=bootstrap_estimators, random_state=random_state)
    print("Combined Scores:")
    print(combined_scores)