import numpy as np
from scipy.sparse import csr_matrix

def make_multilabel_classification(n_samples, n_features, n_classes, n_labels, document_length, p_c=None, p_w_c=None):
    # Generate class prior probabilities if not provided
    if p_c is None:
        p_c = np.random.dirichlet([1] * n_classes)
    
    # Generate feature conditional probabilities if not provided
    if p_w_c is None:
        p_w_c = {i: np.random.dirichlet([1] * n_features) for i in range(n_classes)}
    
    # Initialize arrays to hold samples and labels
    X = np.zeros((n_samples, n_features))
    Y = np.zeros((n_samples, n_classes), dtype=int)
    
    for i in range(n_samples):
        # Sample a class according to the class prior probabilities
        c = np.random.choice(n_classes, p=p_c)
        
        # Sample labels according to the class-conditional label probabilities
        labels = np.random.choice(n_classes, size=n_labels, replace=False, p=p_c)
        
        # Sample document length from a Poisson distribution with mean document_length
        doc_length = np.random.poisson(document_length)
        
        # Sample features from the multinomial distribution given the class
        features = np.random.multinomial(doc_length, p_w_c[c])
        
        # Store the sample and labels
        X[i, :] = features
        Y[i, labels] = 1
    
    return X, Y, p_c, p_w_c

if __name__ == "__main__":
    X, Y, p_c, p_w_c = make_multilabel_classification(n_samples=10, n_features=5, n_classes=3, n_labels=2, document_length=10)
    print("X:", X)
    print("Y:", Y)
    print("Class Prior Probabilities (p_c):", p_c)
    print("Feature Conditional Probabilities (p_w_c):", p_w_c)