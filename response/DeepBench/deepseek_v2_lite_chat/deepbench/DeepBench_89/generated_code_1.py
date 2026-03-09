import numpy as np
from sklearn.decomposition import NMF

def dict_learning(X, n_components, alpha, max_iter=100, tol=1e-8, method='lars', n_jobs=None, dict_init=None, code_init=None, callback=None, verbose=False, random_state=None, return_n_iter=False, positive_dict=False, positive_code=False, method_max_iter=1000):
    if n_jobs is None:
        n_jobs = max(1, -1 * (np.size(X, 0) < 50))

    # Initialize dictionaries and codes
    if dict_init is not None:
        self.dictionary = dict_init
    else:
        self.dictionary = NMF(n_components=n_components, init=dict_init, alpha=alpha, max_iter=method_max_iter, random_state=random_state)

    if code_init is not None:
        self.code = code_init
    else:
        self.code = NMF(n_components=n_components, init=code_init, alpha=alpha, max_iter=method_max_iter, random_state=random_state)

    # Fit the model
    self.dictionary.fit(X, self.code)

    # Calculate errors
    errors = []
    for i in range(max_iter):
        if callback is not None and i % 5 == 0:
            callback(i)

        self.code.n_iter_ = max_iter
        new_dictionary, new_code = self.dictionary.fit_transform(X)
        if positive_dict:
            new_dictionary = np.maximum(new_dictionary, 0)
        if positive_code:
            new_code = np.maximum(new_code, 0)

        error = np.linalg.norm(new_dictionary - self.dictionary.components_)
        errors.append(error)

        if error < tol:
            break

        self.dictionary, self.code = new_dictionary, new_code

    # Return results
    if return_n_iter:
        return self.dictionary, self.code, errors
    else:
        return self.dictionary, self.code

# Run example
if __name__ == "__main__":
    import random
    X = np.random.rand(100, 10)
    n_components = 5
    alpha = 1
    max_iter = 1000
    tol = 1e-8

    # Call the function
    result = dict_learning(X, n_components, alpha, max_iter, tol, callback=lambda i: print(f"Iteration {i}, Error: {result[2][i]}"))

    # Print results
    print("Dictionary:", result[0])
    print("Sparse code:", result[1])
    print("Errors:", result[2])