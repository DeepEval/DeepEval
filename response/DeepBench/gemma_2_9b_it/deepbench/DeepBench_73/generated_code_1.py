import numpy as np
from scipy.sparse import csr_matrix

def make_multilabel_classification(n_samples, n_features, n_classes, n_labels, document_length=None, p_c=None, p_w_c=None):
  if p_c is None:
    p_c = np.ones(n_classes) / n_classes
  if p_w_c is None:
    p_w_c = np.ones((n_classes, n_features)) / n_features
  
  X = np.random.rand(n_samples, n_features)
  
  # while True:
  Y = np.random.choice(n_classes, size=(n_samples, n_labels), p=p_c)
  valid = np.all(np.sum(Y, axis=1) == n_labels, axis=0)
  # if np.any(valid):
  #   break

  if document_length is not None:
    X = X[:, :document_length]

  return X, Y, p_c, p_w_c

if __name__ == "__main__":
  n_samples = 10
  n_features = 20
  n_classes = 5
  n_labels = 3
  
  X, Y, p_c, p_w_c = make_multilabel_classification(n_samples, n_features, n_classes, n_labels)

  print("Samples (X):\n", X)
  print("Label Sets (Y):\n", Y)
  print("Prior Class Probabilities (p_c):\n", p_c)
  print("Conditional Probabilities (p_w_c):\n", p_w_c)