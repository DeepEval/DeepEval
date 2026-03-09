import numpy as np

def make_hastie_10_2(n_samples=1000, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    
    X = np.random.randn(n_samples, 10)
    y = np.sign(X[:, 0] * X[:, 1] + X[:, 2] * X[:, 3] + X[:, 4] * X[:, 5] +
                X[:, 6] * X[:, 7] + X[:, 8] * X[:, 9])
    return X, y

if __name__ == "__main__":
    X, y = make_hastie_10_2(n_samples=5, random_state=42)
    print("Features (X):\n", X)
    print("Labels (y):\n", y)