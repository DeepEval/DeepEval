import numpy as np
from sklearn.utils import extmath
from scipy.sparse import csc_matrix
from sklearn.datasets import make_sparse_coded_signal
from sklearn.linear_model import Lasso

def dict_learning_online(X, n_components=2, alpha=1, max_iter=100, return_code=True, dict_init=None, callback=None, batch_size=256, verbose=False, shuffle=True, n_jobs=1, method='lars', random_state=None, positive_dict=False, positive_code=False, method_max_iter=1000, tol=1e-3, max_no_improvement=10, **kwargs):
    if dict_init is None:
        dict_init = extmath.randomized_svd(X, n_components, random_state=random_state)

    code = None
    dictionary = dict_init
    n_iter = 0
    cost_sum = 0

    U = np.dot(X, dictionary)
    U_norm = np.linalg.norm(U, ord=1, axis=0)
    U /= U_norm

    method_instance = Lasso(alpha=alpha, max_iter=method_max_iter, tol=tol, random_state=random_state)

    for k in range(n_components):
        U_part = U[:, k]
        dictionary_part = dictionary[:, k]

        if callback:
            callback(n_iter, cost_sum, U, dictionary)

        U_part_norm = np.linalg.norm(U_part, ord=1, axis=0)
        U_part /= U_part_norm

        dictionary_part /= np.linalg.norm(dictionary_part)

        U_part_part = csc_matrix(U_part)
        dictionary_part_part = csc_matrix(dictionary_part)

        U_part, resid, rank, s = method_instance.fit(U_part_part, X, method='lasso', fit_intercept=False)

        dictionary[:, k], cost, _, _ = method_instance.coordinate_descent(dictionary_part_part, X, fit_intercept=False)

        dictionary[:, k] /= np.linalg.norm(dictionary[:, k])

        cost_sum += cost
        n_iter += 1

        if n_iter >= max_iter or cost < 1e-5:
            break

        if n_jobs == -1:
            U_part = U_part_part.todense()
            dictionary[:, k] = dictionary_part_part.todense()

    if return_code:
        return U_part, dictionary
    else:
        return dictionary

if __name__ == "__main__":
    # Generate sample data
    n_samples, n_features = 100, 10
    X, code, dictionary = make_sparse_coded_signal(n_samples, n_features, density=0.5)

    # Run the function
    V, U = dict_learning_online(X, n_components=2, alpha=1, max_iter=100, return_code=True, dict_init=None, shuffle=True, verbose=True)

    # Print results
    print("Dictionary V:", V.todense())
    print("Sparse code U:", U)