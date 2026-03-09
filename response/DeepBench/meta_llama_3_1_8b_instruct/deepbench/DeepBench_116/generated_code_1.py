import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
from scipy.linalg import norm
from sklearn.utils.extmath import safe_sparse_dot

def orthogonal_mp_gram(X, y, n_nonzero_coefs=None, tol=None, precompute=False, copy_X=True, return_path=False, return_n_iter=False):
    n_samples, n_features = X.shape
    n_targets = 1 if np.isscalar(y) else y.shape[1]
    
    if n_nonzero_coefs is None:
        n_nonzero_coefs = int(0.1 * n_features)
        
    if n_nonzero_coefs > n_features:
        raise ValueError("n_nonzero_coefs cannot exceed the number of features")
        
    if precompute:
        if isinstance(precompute, bool) and not precompute:
            precompute = 'auto'
            
        if precompute == 'auto':
            precompute = X.shape[0] >= 1000 or X.shape[1] >= 1000
        
        if precompute:
            XTX = X.T.dot(X)
            if n_targets > 1:
                XTy = np.dot(X.T, y)
            else:
                XTy = np.dot(X.T, y[:, None])
        else:
            XTX = None
            XTy = None
            
    residual = y.copy()
    support = np.zeros(n_features, dtype=bool)
    coefficients = np.zeros((n_samples, n_features))
    n_iter = 0
    
    while n_iter < n_nonzero_coefs and n_iter < n_features:
        if XTX is None:
            XTX_residual = safe_sparse_dot(X.T, residual, dense_output=True)
        else:
            XTX_residual = XTX.dot(residual)
            
        if n_targets > 1:
            k = np.argmax(np.abs(XTX_residual))
        else:
            k = np.argmax(np.abs(XTX_residual[0]))
            
        if XTX is None:
            Xk = X[:, k]
        else:
            Xk = X[:, k]
            
        if XTX is None:
            alpha = np.dot(Xk.T, residual) / np.dot(Xk.T, Xk)
        else:
            alpha = np.dot(Xk.T, XTy) / np.dot(Xk.T, XTX.dot(Xk))
            
        if XTX is None:
            residual -= alpha * Xk
        else:
            residual -= alpha * XTX.dot(Xk)
            
        support[k] = True
        coefficients[:, k] = alpha
        
        n_iter += 1
        
    if tol is not None:
        if norm(residual) > tol:
            raise ValueError("Failed to converge")
            
    if return_path:
        return coefficients
    elif return_n_iter:
        return coefficients, n_iter
    else:
        return coefficients[:, support]

if __name__ == "__main__":
    np.random.seed(0)
    X = np.random.rand(100, 1000)
    y = np.random.rand(100)
    coefficients = orthogonal_mp_gram(X, y, n_nonzero_coefs=10, return_path=True)
    print(coefficients)
    
    coefficients, n_iter = orthogonal_mp_gram(X, y, n_nonzero_coefs=10, return_path=False, return_n_iter=True)
    print(coefficients)
    print(n_iter)