import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import RandomForestClassifier

def function_name(scores, n_buckets=5, method='static', bootstrap_estimators=False, random_state=None):
    if method == 'static':
        n_samples = scores.shape[0]
        n_estimators = scores.shape[1]
        scores = scores.reshape(-1)
        bins = np.histogram_bin_edges(np.argpartition(scores, -n_buckets)[-n_buckets:], bins=n_buckets)[1]
        group_scores = np.digitize(scores, bins)
        combined_scores = np.mean(scores[group_scores], axis=0)
    elif method == 'dynamic':
        raise NotImplementedError("Dynamic subgroup building is not implemented.")
    else:
        raise ValueError("Method must be either 'static' or 'dynamic'.")
    
    return combined_scores

if __name__ == "__main__":
    np.random.seed(0)
    
    n_samples = 100
    n_estimators = 5
    n_buckets = 5
    
    # Generate scores from two estimators
    scores = np.random.rand(n_samples, n_estimators)
    
    # Generate outlier scores
    outlier_scores = np.random.choice([-1, 1], size=(n_samples, n_estimators))
    scores += outlier_scores
    
    # Create a random state
    random_state = np.random.RandomState(0)
    
    # Call the function
    combined_scores = function_name(scores, n_buckets=n_buckets, method='static', bootstrap_estimators=False, random_state=random_state)
    
    # Print the results
    print("Combined scores:", combined_scores)