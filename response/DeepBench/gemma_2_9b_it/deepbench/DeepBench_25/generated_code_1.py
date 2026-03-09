import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

def aom(scores, n_buckets=5, method='static', bootstrap_estimators=False, random_state=None):
  n_samples, n_estimators = scores.shape
  if method == 'static':
    bucket_size = n_estimators // n_buckets
    buckets = [slice(i * bucket_size, (i + 1) * bucket_size) for i in range(n_buckets)]
  elif method == 'dynamic':
    if bootstrap_estimators:
      indices = np.random.choice(n_estimators, size=n_estimators, replace=True)
    else:
      indices = np.arange(n_estimators)
    buckets = np.array_split(indices, n_buckets)
  else:
    raise ValueError("Invalid method. Choose from 'static' or 'dynamic'.")

  combined_scores = np.zeros(n_samples)
  for i in range(n_samples):
    for j in range(n_buckets):
      bucket_scores = scores[i, buckets[j]]
      combined_scores[i] = max(combined_scores[i], np.mean(bucket_scores))
  return combined_scores

if __name__ == "__main__":
    X, y = make_classification(n_samples=100, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    clf1 = RandomForestClassifier(random_state=42)
    clf2 = RandomForestClassifier(random_state=42)
    clf3 = RandomForestClassifier(random_state=42)

    clf1.fit(X_train, y_train)
    clf2.fit(X_train, y_train)
    clf3.fit(X_train, y_train)

    scores_clf1 = clf1.decision_function(X_test)
    scores_clf2 = clf2.decision_function(X_test)
    scores_clf3 = clf3.decision_function(X_test)

    scores = np.stack((scores_clf1, scores_clf2, scores_clf3), axis=1)
    
    combined_scores = aom(scores, n_buckets=3, method='static')

    print(combined_scores)