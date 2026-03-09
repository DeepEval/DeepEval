import numpy as np
from sklearn.decomposition import DictionaryLearning
from sklearn.utils import check_random_state

def dict_learning(X, n_components, alpha=1, max_iter=100, tol=1e-8, method='lars', n_jobs=None, dict_init=None, code_init=None, callback=None, verbose=False, random_state=None, return_n_iter=False, positive_dict=False, positive_code=False, method_max_iter=1000):
    random_state = check_random_state(random_state)
    model = DictionaryLearning(n_components=n_components, alpha=alpha, max_iter=max_iter, tol=tol, method=method, n_jobs=n_jobs, dict_init=dict_init, code_init=code_init, callback=callback, verbose=verbose, random_state=random_state, return_n_iter=return_n_iter, positive_dict=positive_dict, positive_code=positive_code, method_max_iter=method_max_iter)
    code, dictionary = model.fit_transform(X)
    errors = model.reconstruction_err_
    if return_n_iter:
        return code, dictionary, errors, model.n_iter_
    else:
        return code, dictionary, errors

if __name__ == "__main__":
    # Create a sample data matrix
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    n_samples, n_features = X.shape
    n_components = 2
    alpha = 0.1

    # Call the function
    code, dictionary, errors = dict_learning(X, n_components, alpha=alpha)

    # Print the results
    print("Dictionary:\n", dictionary)
    print("Sparse code:\n", code)
    print("Errors:\n", errors)