import numpy as np
from sklearn.utils import check_random_state
from sklearn.utils.extmath import randomized_svd
from sklearn.linear_model import lars_path, Lasso
from sklearn.decomposition import DictionaryLearning
import joblib
import time

def dict_learning_online(X, n_components, alpha, max_iter=100, return_code=True, dict_init=None, callback=None, batch_size=256, verbose=False, shuffle=True, n_jobs=None, method='lars', random_state=None, positive_dict=False, positive_code=False, method_max_iter=1000, tol=1e-3, max_no_improvement=10):
    n_samples, n_features = X.shape
    n_components = int(n_components)

    if dict_init is None:
        dict_init = randomized_svd(X, n_components=n_components, random_state=random_state)[0].T

    if n_jobs is None:
        n_jobs = 1

    n_iter = 0
    batch_idx = 0
    n_batches = int(np.ceil(n_samples / batch_size))
    cost_history = [np.inf]
    max_cost = np.inf
    max_idx = 0
    start_time = time.time()
    random_state = check_random_state(random_state)

    while n_iter < max_iter and batch_idx < n_batches:
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, n_samples)
        batch_X = X[batch_start:batch_end, :]

        if shuffle:
            indices = np.arange(batch_start, batch_end)
            random_state.shuffle(indices)
            batch_X = batch_X[indices, :]

        if method == 'lars':
            U_path = lars_path(batch_X, dict_init, method='lasso', alpha=alpha, copy_X=True, verbose=verbose, Xy=None, precompute=False, eps=1e-14, n_jobs=n_jobs, positive=positive_dict, return_ny=True, return_path=True, method_max_iter=method_max_iter)
            V = U_path[:, -1]
            U = U_path[0, :]
        elif method == 'cd':
            lasso = Lasso(alpha=alpha, fit_intercept=False, positive=positive_dict, random_state=random_state, max_iter=method_max_iter)
            lasso.fit(batch_X, np.ones(batch_X.shape[0]))
            U = lasso.coef_.reshape(-1, 1)
            V = lasso.predict(batch_X).reshape(-1, 1)

        if return_code:
            U = U.T

        if positive_dict:
            V = np.maximum(V, 0)

        if positive_code:
            U = np.maximum(U, 0)

        new_cost = 0.5 * np.linalg.norm(batch_X - U * V) ** 2 + alpha * np.sum(np.abs(U))
        cost_history.append(new_cost)

        if n_iter >= 1:
            if np.linalg.norm(U - dict_init) < tol:
                if verbose:
                    print('Convergence detected due to tolerance')
                break
            elif len(cost_history) > max_no_improvement + 1 and new_cost > cost_history[max_idx]:
                if verbose:
                    print('Convergence detected due to no improvement')
                break
            else:
                max_idx = len(cost_history) - 1
                max_cost = new_cost

        dict_init = U
        batch_idx += 1
        n_iter += 1

    if return_code:
        code = U
    else:
        code = None

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Total elapsed time: {elapsed_time} seconds')

    return code, dict_init, n_iter, cost_history

if __name__ == "__main__":
    np.random.seed(0)
    X = np.random.rand(100, 10)
    code, dict_init, n_iter, cost_history = dict_learning_online(X, n_components=5, alpha=1, max_iter=10, return_code=True, batch_size=10)
    print(f'Code: {code}')
    print(f'Dictionary: {dict_init}')
    print(f'Number of iterations: {n_iter}')
    print(f'Cost history: {cost_history}')