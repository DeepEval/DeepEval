import numpy as np
from sklearn.utils import check_random_state

def aom(scores, n_buckets=5, method='static', bootstrap_estimators=False, random_state=None):
    n_samples, n_estimators = scores.shape
    random_state = check_random_state(random_state)
    
    if method =='static':
        indices = np.argsort(scores, axis=1)
    elif method == 'dynamic':
        if bootstrap_estimators:
            estimator_indices = random_state.randint(0, n_estimators, size=(n_samples,
  n_buckets))
        else:
            estimator_indices = np.argsort(scores, axis=1)[:, :n_buckets]
        indices = np.argsort(scores[estimator_indices], axis=1)
    else:
        raise ValueError("Invalid method. Choose'static' or 'dynamic'.")
    
    max_scores = np.max(scores[:, indices], axis=1)
    combined_scores = np.mean(max_scores)
    
    return np.full((n_samples,), combined_scores)

if __name__ == "__main__":
    n_samples = 100
    n_estimators = 10
    scores = np.random.rand(n_samples, n_estimators)
    n_buckets = 5
    method ='static'
    bootstrap_estimators = False
    random_state = 42
    
    combined_scores = aom(scores, n_buckets, method, bootstrap_estimators, random_state)
    print(combined_scores)