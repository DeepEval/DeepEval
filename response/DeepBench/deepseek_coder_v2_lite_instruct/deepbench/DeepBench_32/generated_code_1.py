from sklearn.datasets import make_classification

def make_hastie_10_2(n_samples=12000, random_state=None):
    # Generate a synthetic binary classification dataset
    X, y = make_classification(n_samples=n_samples, n_features=20, random_state=random_state)
    return X, y

if __name__ == "__main__":
    # Create sample input values
    X, y = make_hastie_10_2(n_samples=10)
    
    # Print the results
    print("Features:\n", X)
    print("Labels:\n", y)