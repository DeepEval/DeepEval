import numpy as np
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.exceptions import ConvergenceWarning
import warnings

def dict_learning(X, n_components, alpha, max_iter=100, tol=1e-8, method='lars', n_jobs=None, dict_init=None, code_init=None, callback=None, verbose=False, random_state=None, return_n_iter=False, positive_dict=False, positive_code=False, method_max_iter=1000):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)

    # Initialize dictionary learning model
    model = MiniBatchDictionaryLearning(n_components=n_components, alpha=alpha, max_iter=max_iter, tol=tol, method=method, n_jobs=n_jobs, dict_init=dict_init, code_init=code_init, callback=callback, verbose=verbose, random_state=random_state, positive_dict=positive_dict, positive_code=positive_code, batch_size=min(300, X.shape[0]), max_iter_method=method_max_iter)

    # Fit the model to the data
    model.fit(X)

    # Get the learned dictionary and code
    dictionary = model.components_
    code = model.code_

    # Get the errors at each iteration
    errors = model._check_params()

    # If return_n_iter is True, return the number of iterations
    if return_n_iter:
        n_iter = model.n_iter_
        return code, dictionary, errors, n_iter
    else:
        return code, dictionary, errors

if __name__ == "__main__":
    # Create sample input values
    n_samples, n_features = 100, 10
    X = np.random.rand(n_samples, n_features)

    # Call the function and print the results
    code, dictionary, errors = dict_learning(X, n_components=5, alpha=0.1)
    print("Sparse code:")
    print(code)
    print("Dictionary:")
    print(dictionary)
    print("Errors:")
    print(errors)