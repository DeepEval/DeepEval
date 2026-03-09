import numpy as np
from sklearn.decomposition import DictionaryLearning
from sklearn.utils.extmath import randomized_svd

def dict_learning_online(X, n_components=2, alpha=1, max_iter=100, return_code=True, dict_init=None, callback=None, batch_size=256, verbose=False, shuffle=True, n_jobs=None, method='lars', random_state=None, positive_dict=False, positive_code=False, method_max_iter=1000, tol=1e-3, max_no_improvement=10):
    if dict_init is None:
        _, n_features = X.shape
        dict_init = randomized_svd(X, n_components, random_state=random_state)[0]

    if method == 'lars':
        from sklearn.linear_model import Lars
        lasso = Lars(fit_intercept=False, positive=positive_code, max_iter=method_max_iter)
    elif method == 'cd':
        from sklearn.linear_model import Lasso
        lasso = Lasso(fit_intercept=False, positive=positive_code, max_iter=method_max_iter, tol=tol)
    else:
        raise ValueError("Invalid method specified. Use 'lars' or 'cd'.")

    n_samples, n_features = X.shape
    code = np.zeros((n_samples, n_components))
    dictionary = dict_init

    for _ in range(max_iter):
        if shuffle:
            indices = np.random.permutation(n_samples)
        else:
            indices = np.arange(n_samples)

        for start in range(0, n_samples, batch_size):
            batch_indices = indices[start:start + batch_size]
            X_batch = X[batch_indices]

            lasso.fit(dictionary, X_batch)
            code[batch_indices] = lasso.coef_

        if callback is not None:
            callback(dictionary, code)

        # Update dictionary using SVD on the mean of X and the code
        U, S, Vt = randomized_svd(X - code @ dictionary, n_components, random_state=random_state)
        dictionary = U * S[:, np.newaxis]

        # Early stopping criteria
        if tol > 0:
            diff = np.linalg.norm(dictionary - dictionary_old)
            if diff < tol:
                break

        if max_no_improvement is not None:
            if max_no_improvement > 0:
                if max_no_improvement == 1:
                    if np.linalg.norm(dictionary - dictionary_old) < tol:
                        max_no_improvement -= 1
                else:
                    if np.linalg.norm(dictionary - dictionary_old) < tol:
                        max_no_improvement -= 1

    if return_code:
        return code, dictionary
    else:
        return dictionary

if __name__ == "__main__":
    X = np.random.rand(100, 20)
    dictionary, code = dict_learning_online(X, n_components=5, max_iter=10)
    print("Dictionary:\n", dictionary)
    print("Code:\n", code)