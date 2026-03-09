import numpy as np

def orthogonal_mp_gram(X, y, n_nonzero_coefs=None, tol=None, precompute='auto', copy_X=True, return_path=False, return_n_iter=False):
    X = np.array(X, copy=copy_X)
    y = np.array(y)
    
    n_samples, n_features = X.shape
    if n_nonzero_coefs is None:
        n_nonzero_coefs = max(1, n_features // 10)
    if n_nonzero_coefs > n_features:
        raise ValueError("n_nonzero_coefs must not exceed the number of features.")
    
    if precompute == 'auto':
        precompute = n_samples > n_features
    
    if precompute:
        gram = np.dot(X.T, X)
    
    coefs = np.zeros(n_features)
    residual = y.copy()
    selected = []
    iter_path = []
    
    for iteration in range(n_nonzero_coefs):
        if precompute:
            correlations = np.dot(X.T, residual)
        else:
            correlations = np.array([np.dot(X[:, j], residual) for j in range(n_features)])
        
        best_index = np.argmax(np.abs(correlations))
        selected.append(best_index)
        
        X_selected = X[:, selected]
        coefs_selected = np.linalg.lstsq(X_selected, y, rcond=None)[0]
        coefs[selected] = coefs_selected
        
        residual = y - np.dot(X_selected, coefs_selected)
        
        if tol is not None and np.dot(residual, residual) < tol:
            break
        
        if return_path:
            iter_path.append(coefs.copy())
    
    if return_n_iter:
        return coefs, iteration + 1
    if return_path:
        return coefs, iter_path
    return coefs

if __name__ == "__main__":
    np.random.seed(0)
    X = np.random.rand(100, 20)
    y = np.random.rand(100)
    coefs = orthogonal_mp_gram(X, y, n_nonzero_coefs=5, tol=1e-6, precompute=True, return_n_iter=True)
    print("Coefficients:", coefs[0])
    print("Number of iterations:", coefs[1])