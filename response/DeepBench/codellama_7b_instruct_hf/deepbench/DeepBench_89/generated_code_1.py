import numpy as np
from sklearn.utils import check_array
from sklearn.linear_model import Lasso
from sklearn.decomposition import DictionaryLearning

def dict_learning(X, n_components, alpha, max_iter=100, tol=1e-8, method='lars', n_jobs=None, dict_init=None, code_init=None, callback=None, verbose=False, random_state=None, return_n_iter=False, positive_dict=False, positive_code=False, method_max_iter=1000):
    # Check input parameters
    X = check_array(X, ensure_2d=True)
    if n_components is None:
        n_components = min(X.shape)
    if alpha is None:
        alpha = 1.0
    if max_iter is None:
        max_iter = 100
    if tol is None:
        tol = 1e-8
    if method is None:
        method = 'lars'
    if n_jobs is None:
        n_jobs = None
    if dict_init is None:
        dict_init = None
    if code_init is None:
        code_init = None
    if callback is None:
        callback = None
    if verbose is None:
        verbose = False
    if random_state is None:
        random_state = None
    if return_n_iter is None:
        return_n_iter = False
    if positive_dict is None:
        positive_dict = False
    if positive_code is None:
        positive_code = False
    if method_max_iter is None:
        method_max_iter = 1000
    
    # Initialize the dictionary and code
    if dict_init is not None and code_init is not None:
        dictionary = dict_init
        code = code_init
    else:
        dictionary = np.random.rand(X.shape[0], n_components)
        code = np.random.rand(n_components, X.shape[1])
    
    # Initialize the error vector
    errors = np.zeros(max_iter)
    
    # Iterate until convergence
    for i in range(max_iter):
        # Compute the dot product between the dictionary and the code
        dot_product = np.dot(dictionary, code)
        
        # Compute the residuals
        residuals = X - dot_product
        
        # Compute the L1 regularization term
        l1_reg = np.sum(np.abs(dictionary))
        
        # Compute the L2 regularization term
        l2_reg = np.sum(np.square(dictionary))
        
        # Compute the total regularization term
        regularization = alpha * (l1_reg + l2_reg)
        
        # Compute the loss function
        loss = np.sum(residuals ** 2) + regularization
        
        # Compute the gradient of the loss function with respect to the dictionary
        gradient = 2 * np.dot(residuals.T, code) + 2 * alpha * dictionary
        
        # Update the dictionary
        dictionary -= np.dot(residuals.T, code) / l1_reg
        
        # Update the code
        code -= np.dot(dictionary.T, residuals) / l2_reg
        
        # Compute the error
        error = np.sum(residuals ** 2) / (n_samples - n_components)
        
        # Update the error vector
        errors[i] = error
        
        # Check for convergence
        if error < tol:
            break
    
    # Return the results
    if return_n_iter:
        return code, dictionary, errors, i
    else:
        return code, dictionary, errors

if __name__ == "__main__":
    # Generate sample input values
    n_samples, n_features = 100, 10
    X = np.random.rand(n_samples, n_features)
    n_components = 5
    alpha = 1.0
    
    # Call the function
    code, dictionary, errors = dict_learning(X, n_components, alpha)
    
    # Print the results
    print("Code:", code)
    print("Dictionary:", dictionary)
    print("Errors:", errors)