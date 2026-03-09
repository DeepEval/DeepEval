import numpy as np

def orthogonal_mp_gram(X, y, n_nonzero_coefs=None, tol=None, precompute='auto', copy_X=True, return_path=False, return_n_iter=False):
    n_samples, n_features = X.shape
    if n_nonzero_coefs is None:
        n_nonzero_coefs = int(n_features * 0.1)
    elif n_nonzero_coefs > n_features:
        raise ValueError("The number of non-zero coefficients cannot exceed the number of features.")
    
    # Initialize coefficients with zeros
    coefficients = np.zeros(n_features)
    
    # Gram matrix
    gram_matrix = X.T.dot(X)
    
    if precompute == 'auto':
        # Precompute Gram matrix if needed
        if n_samples > 10000 or n_features > 1000:
            gram_matrix = gram_matrix.todense()
        else:
            gram_matrix = gram_matrix.tocoo()
            gram_matrix = gram_matrix.astype(float, copy=False)
    
    # Iterative process
    for i in range(n_features):
        if tol is not None:
            residuals = np.abs(y - X.dot(coefficients))
            max_residual = np.max(residuals)
            if max_residual < tol:
                break
        
        # Column of X with max correlation with residuals
        max_correlation = np.abs(X[:, i]).max()
        max_corr_indices = np.where(np.abs(X[:, i]) == max_correlation)[0][0]
        
        # Update coefficients
        coefficients[i] = max_correlation
        if return_path:
            # Store the indices of the selected features in each iteration
            path.append(max_corr_indices)
        
        # Update residuals and X
        y_resid = y - X.dot(coefficients)
        X_selected = X[:, max_corr_indices]
        gram_matrix_selected = X_selected.T.dot(X_selected)
        gram_matrix = gram_matrix - gram_matrix_selected
        gram_matrix = gram_matrix.tocoo()
        
        if copy_X:
            X = X.copy()
        X[max_corr_indices] = 0
    
    if return_path:
        return coefficients, np.array(path)
    elif return_n_iter:
        return len(path)
    else:
        return coefficients

if __name__ == "__main__":
    # Sample input values
    X = np.random.rand(100, 10)  # Random data matrix
    y = np.random.rand(100,)      # Random labels
    
    # Call the function
    coefficients, path = orthogonal_mp_gram(X, y, n_nonzero_coefs=5, tol=1e-6, return_path=True)
    
    # Print the results
    print("Coefficients:", coefficients)
    print("Selected path indices:", path)