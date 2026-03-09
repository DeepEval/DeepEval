import numpy as np

def make_hastie_10_2(n_samples=100):
    X = np.random.rand(n_samples, 2)
    y = (X[:, 0] + X[:, 1]) > 0.5

    return X, y.astype(int)

if __name__ == "__main__":
    X, y = make_hastie_10_2()
    print(X)
    print(y)