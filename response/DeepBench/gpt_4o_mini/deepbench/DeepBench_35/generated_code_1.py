import numpy as np
from sklearn.utils.extmath import randomized_svd
from sklearn.linear_model import Lasso
from sklearn.utils import shuffle as shuffle_data

def dict_learning_online(X, n_components=2, alpha=1, max_iter=100, return_code=True,
                         dict_init=None, callback=None, batch_size=256, verbose=False,
                         shuffle=True, n_jobs=None, method='lars', random_state=None,
                         positive_dict=False, positive_code=False, method_max_iter=1000,
                         tol=1e-3, max_no_improvement=10):
    n_samples, n_features = X.shape
    if n_components is None:
        n_components = n_features
    
    if dict_init is None:
        U, _, Vt = randomized_svd(X, n_components, random_state=random_state)
        dictionary = Vt.T
    else:
        dictionary = dict_init.copy()

    if shuffle:
        X = shuffle_data(X, random_state=random_state)

    n_iter = 0
    no_improvement_count = 0
    best_cost = np.inf
    
    while n_iter < max_iter:
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            X_batch = X[start:end]

            # Normalize the dictionary
            norms = np.linalg.norm(dictionary, axis=1, keepdims=True)
            if np.any(norms == 0):
                continue
            dictionary /= norms
            
            # Solve for U (sparse code)
            lasso = Lasso(alpha=alpha, fit_intercept=False, max_iter=method_max_iter, tol=tol)
            lasso.fit(dictionary.T, X_batch.T)
            U_batch = lasso.coef_.T
            
            if positive_code:
                U_batch[U_batch < 0] = 0
            
            # Update dictionary using U
            dictionary += np.dot(U_batch.T, X_batch) / batch_size
            
            if positive_dict:
                dictionary[dictionary < 0] = 0

            # Compute cost
            cost = 0.5 * np.sum((X_batch - np.dot(U_batch, dictionary.T)) ** 2) + alpha * np.sum(np.abs(dictionary))
            
            if cost < best_cost:
                best_cost = cost
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            if max_no_improvement is not None and no_improvement_count >= max_no_improvement:
                break
            
        n_iter += 1
        if callback is not None:
            callback()

        if verbose:
            print(f"Iteration {n_iter}: cost={cost}")

    if return_code:
        return U_batch, dictionary
    else:
        return dictionary

if __name__ == "__main__":
    X = np.random.rand(100, 20)
    U, V = dict_learning_online(X, n_components=5, alpha=0.5, max_iter=10, verbose=True)
    print("Sparse Code (U):\n", U)
    print("Dictionary (V):\n", V)