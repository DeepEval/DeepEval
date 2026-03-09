import numpy as np
from sklearn.decomposition import DictionaryLearning

def dict_learning(X, n_components, alpha, max_iter=100, tol=1e-8,
                  method='lars', n_jobs=None, dict_init=None,
                  code_init=None, callback=None, verbose=False,
                  random_state=None, return_n_iter=False,
                  positive_dict=False, positive_code=False):
    
    dl = DictionaryLearning(n_components=n_components, alpha=alpha, 
                            max_iter=max_iter, tol=tol, 
                            transform_algorithm=method, n_jobs=n_jobs, 
                            dict_init=dict_init, code_init=code_init, 
                            callback=callback, verbose=verbose,
                            random_state=random_state, 
                            positive_dict=positive_dict, 
                            positive_code=positive_code)

    code = dl.fit_transform(X)
    dictionary = dl.components_
    errors = dl.error_

    if return_n_iter:
        return code, dictionary, errors, dl.n_iter_
    else:
        return code, dictionary, errors

if __name__ == "__main__":
    np.random.seed(0)
    X = np.random.rand(10, 5)  # 10 samples, 5 features
    n_components = 3
    alpha = 1.0

    code, dictionary, errors = dict_learning(X, n_components, alpha)

    print("Sparse Code:\n", code)
    print("Dictionary:\n", dictionary)
    print("Errors:\n", errors)