from sklearn.decomposition import dict_learning

def dict_learning(X, n_components, alpha, max_iter=100, tol=1e-8, method='lars',
                 n_jobs=None, dict_init=None, code_init=None, callback=None,
                 verbose=False, random_state=None, return_n_iter=False,
                 positive_dict=False, positive_code=False, method_max_iter=1000):
    """
    Solves a dictionary learning matrix factorization problem.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Input data matrix.
    n_components : int
        Number of dictionary atoms to extract.
    alpha : float
        Sparsity controlling parameter.
    max_iter : int, default=100
        Maximum number of iterations to perform.
    tol : float, default=1e-8
        Tolerance for the stopping condition.
    method : str, default='lars'
        Method used for solving the problem.
    n_jobs : int, default=None
        Number of parallel jobs to run.
    dict_init : array-like, shape (n_components, n_features), default=None
        Initial value for the dictionary for warm restart scenarios.
    code_init : array-like, shape (n_samples, n_components), default=None
        Initial value for the sparse code for warm restart scenarios.
    callback : callable, default=None
        Callable that gets invoked every five iterations.
    verbose : bool, default=False
        Verbosity of the procedure.
    random_state : int, default=None
        Used for randomly initializing the dictionary.
    return_n_iter : bool, default=False
        Whether or not to return the number of iterations.
    positive_dict : bool, default=False
        Whether to enforce positivity when finding the dictionary.
    positive_code : bool, default=False
        Whether to enforce positivity when finding the code.
    method_max_iter : int, default=1000
        Maximum number of iterations to perform.

    Returns
    -------
    code : array-like, shape (n_samples, n_components)
        Sparse code factor in the matrix factorization.
    dictionary : array-like, shape (n_components, n_features)
        Dictionary factor in the matrix factorization.
    errors : array-like, shape (max_iter,)
        Vector of errors at each iteration.

    """
    return dict_learning(X, n_components, alpha, max_iter=max_iter, tol=tol, method=method,
                      n_jobs=n_jobs, dict_init=dict_init, code_init=code_init, callback=callback,
                      verbose=verbose, random_state=random_state, return_n_iter=return_n_iter,
                      positive_dict=positive_dict, positive_code=positive_code, method_max_iter=method_max_iter)

if __name__ == "__main__":
    # Sample input values
    X = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    n_components = 2
    alpha = 0.1

    # Call the function and print the results
    code, dictionary, errors = dict_learning(X, n_components, alpha)
    print("Sparse code:", code)
    print("Dictionary:", dictionary)
    print("Errors:", errors)