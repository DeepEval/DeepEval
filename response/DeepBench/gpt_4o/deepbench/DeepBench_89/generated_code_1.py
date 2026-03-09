import numpy as np
from sklearn.decomposition import DictionaryLearning

def dict_learning(X, n_components, alpha, max_iter=100, tol=1e-8, method='lars', n_jobs=None,
                  dict_init=None, code_init=None, callback=None, verbose=False, random_state=None,
                  return_n_iter=False, positive_dict=False, positive_code=False, method_max_iter=1000):
    
    model = DictionaryLearning(
        n_components=n_components,
        alpha=alpha,
        max_iter=max_iter,
        tol=tol,
        method=method,
        n_jobs=n_jobs,
        dict_init=dict_init,
        code_init=code_init,
        verbose=verbose,
        random_state=random_state,
        positive_dict=positive_dict,
        positive_code=positive_code,
        transform_max_iter=method_max_iter
    )

    model.fit(X)
    code = model.transform(X)
    dictionary = model.components_
    errors = model.error_ if hasattr(model, 'error_') else None
    
    if return_n_iter:
        return code, dictionary, errors, model.n_iter_
    else:
        return code, dictionary, errors

if __name__ == "__main__":
    np.random.seed(0)
    X = np.random.rand(10, 8)
    n_components = 5
    alpha = 1
    code, dictionary, errors = dict_learning(X, n_components, alpha, max_iter=100, verbose=True)
    
    print("Sparse Code:\n", code)
    print("Dictionary:\n", dictionary)
    print("Errors:\n", errors)