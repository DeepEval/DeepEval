import numpy as np

def make_hastie_10_2(n_samples=1000, n_features=20, n_classes=2, coef=1):
    """
    Generate data for binary classification as used in Hastie et al. 2009.

    Parameters:
    n_samples: Number of samples to generate
    n_features: Number of features in the data
    n_classes: Number of classes in the target variable
    coef: Coefficient of the polynomial features

    Returns:
    X: Input data
    y: Target variable
    """
    # Generate random data
    X = np.random.rand(n_samples, n_features)
    y = np.random.randint(0, n_classes, n_samples)

    # Add polynomial features
    X = np.hstack((X, np.power(X, coef)))

    # Add noise to the data
    noise = np.random.normal(size=(n_samples, n_features))
    X += noise

    # Split the data into training and testing sets
    train_size = int(0.8 * n_samples)
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]

    return X_train, y_train, X_test, y_test

# Example usage
if __name__ == "__main__":
    X_train, y_train, X_test, y_test = make_hastie_10_2(n_samples=1000, n_features=20, n_classes=2, coef=1)
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)