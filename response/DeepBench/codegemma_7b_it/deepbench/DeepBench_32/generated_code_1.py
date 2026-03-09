from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

def make_hastie_10_2(n_samples=4000, n_features=10, n_informative=3, n_redundant=1, n_repeated=0, n_classes=2, random_state=42):
    X, y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=n_informative, n_redundant=n_redundant, n_repeated=n_repeated, n_classes=n_classes, random_state=random_state)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y

if __name__ == "__main__":
    # Create sample input values
    n_samples = 500
    n_features = 5
    n_informative = 3
    n_redundant = 1
    n_repeated = 0
    n_classes = 2
    random_state = 42

    # Call the function
    X, y = make_hastie_10_2(n_samples, n_features, n_informative, n_redundant, n_repeated, n_classes, random_state)

    # Print the results
    print("X shape:", X.shape)
    print("y shape:", y.shape)