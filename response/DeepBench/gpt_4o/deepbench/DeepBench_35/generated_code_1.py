import numpy as np
from sklearn.utils import check_random_state
from sklearn.utils.extmath import randomized_svd
from sklearn.linear_model import lars_path, Lasso
from sklearn.utils import gen_batches

def dict_learning_online(X, n_components=2, alpha=1, max_iter=100, return_code=True,
                         dict_init=None, callback=None, batch_size=256, verbose=False,
                         shuffle=True, n_jobs=None, method='lars', random_state=None,
                         positive_dict=False, positive_code=False, method_max_iter=1000,
                         tol=1e-3, max_no_improvement=10):
    
    n_samples, n_features = X.shape
    random_state = check_random_state(random_state)
    n_components = n_features if n_components is None else n_components
    
    if dict_init is None:
        _, S, Vt = randomized_svd(X, n_components=n_components, random_state=random_state)
        dictionary = Vt
    else:
        dictionary = dict_init

    if shuffle:
        X = X[random_state.permutation(n_samples)]
    
    batches = list(gen_batches(n_samples, batch_size))
    n_batches = len(batches)
    
    code = np.zeros((n_samples, n_components))
    best_error = np.inf
    no_improvement = 0
    
    for iteration in range(max_iter):
        for batch in batches:
            X_batch = X[batch]
            
            if method == 'lars':
                _, _, coefs = lars_path(X_batch, dictionary, alpha=alpha, method='lasso')
                code[batch] = coefs.T
            elif method == 'cd':
                lasso = Lasso(alpha=alpha, fit_intercept=False, max_iter=method_max_iter,
                              positive=positive_code)
                lasso.fit(dictionary.T, X_batch.T)
                code[batch] = lasso.coef_.T
            
            for k in range(n_components):
                code[:, k] /= np.linalg.norm(code[:, k], ord=2)
            
            new_dict = np.zeros_like(dictionary)
            for k in range(n_components):
                if positive_dict:
                    new_dict[k] = np.maximum(0, np.dot(code[:, k].T, X) / np.dot(code[:, k], code[:, k].T))
                else:
                    new_dict[k] = np.dot(code[:, k].T, X) / np.dot(code[:, k], code[:, k].T)
            
            dictionary = new_dict
            
        error = 0.5 * np.linalg.norm(X - np.dot(code, dictionary), 'fro') ** 2 + alpha * np.sum(np.abs(code))
        
        if verbose:
            print(f"Iteration {iteration + 1}, error: {error}")
        
        if callback is not None:
            callback(dictionary)
        
        if error < best_error - tol:
            best_error = error
            no_improvement = 0
        else:
            no_improvement += 1
        
        if max_no_improvement is not None and no_improvement >= max_no_improvement:
            if verbose:
                print("Convergence reached after no improvement")
            break

    if return_code:
        return code, dictionary, iteration + 1
    else:
        return dictionary, iteration + 1

if __name__ == "__main__":
    X_sample = np.random.rand(100, 20)
    n_components = 5
    alpha = 1.0
    batch_size = 10

    code, dictionary, n_iter = dict_learning_online(X_sample, n_components=n_components, alpha=alpha, 
                                                    batch_size=batch_size, verbose=True)

    print("Final dictionary:\n", dictionary)
    print("Sparse code:\n", code)
    print("Number of iterations:", n_iter)