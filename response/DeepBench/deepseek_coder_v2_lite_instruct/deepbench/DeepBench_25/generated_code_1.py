import numpy as np

def aom(scores, n_buckets=5, method='static', bootstrap_estimators=False, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples, n_estimators = scores.shape
    combined_scores = np.zeros(n_samples)
    
    for i in range(n_samples):
        if method == 'static':
            idx = np.argsort(scores[i, :])
        elif method == 'dynamic':
            if bootstrap_estimators:
                idx = np.random.choice(np.arange(n_estimators), size=n_estimators, replace=True)
            else:
                idx = np.argsort(np.random.rand(n_estimators))
        else:
            raise ValueError("Invalid method specified. Use 'static' or 'dynamic'.")
        
        idx = idx[:n_buckets]
        combined_scores[i] = np.mean(np.max(scores[i, idx], axis=0))
    
    return combined_scores

if __name__ == "__main__":
    scores = np.array([[0.1, 0.2, 0.3, 0.4, 0.5],
                       [0.5, 0.4, 0.3, 0.2, 0.1],
                       [0.3, 0.3, 0.3, 0.3, 0.3]])
    result = aom(scores, n_buckets=3, method='static', bootstrap_estimators=False, random_state=42)
    print(result)