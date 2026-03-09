from typing import Optional
from collections import defaultdict

import numpy as np

from sklearn.utils import check_array, check_random_state
from sklearn.linear_model import lars_path, Lasso
from sklearn.utils.extmath import randomized_svd
from sklearn.exceptions import ConvergenceWarning

def dict_learning_online(
    X: np.ndarray,
    n_components: Optional[int] = None,
    alpha: float = 1.0,
    max_iter: int = 100,
    return_code: bool = True,
    dict_init: Optional[np.ndarray] = None,
    callback: Optional[Callable] = None,
    batch_size: int = 256,
    verbose: bool = False,
    shuffle: bool = True,
    n_jobs: Optional[int] = None,
    method: str = "lars",
    positive_dict: bool = False,
    positive_code: bool = False,
    method_max_iter: int = 1000,
    tol: float = 1e-3,
    max_no_improvement: int = 10,
) -> dict:
    """Solve a dictionary learning matrix factorization problem online.

    This function is the online version of the `dict_learning` function. It
    iteratively updates the dictionary and the code as new data becomes
    available.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data matrix.
    n_components : int or None, default=2
        Number of dictionary atoms to extract. If None, then ``n_components``
        is set to ``n_features``.
    alpha : float, default=1
        Sparsity controlling parameter.
    max_iter : int, default=100
        Maximum number of iterations over the complete dataset before
        stopping independently of any early stopping criterion heuristics.
    return_code : bool, default=True
        Whether to also return the code U or just the dictionary `V`.
    dict_init : ndarray of shape (n_components, n_features), default=None
        Initial values for the dictionary for warm restart scenarios. If
        `None`, the initial values for the dictionary are created with an SVD
        decomposition of the data via :func:`~sklearn.utils.extmath.randomized_svd`.
    callback : callable, default=None
        A callable that gets invoked at the end of each iteration.
    batch_size : int, default=256
        The number of samples to take in each batch.
    verbose : bool, default=False
        To control the verbosity of the procedure.
    shuffle : bool, default=True
        Whether to shuffle the data before splitting it in batches.
    n_jobs : int, default=None
        Number of parallel jobs to run. ``None`` means 1 unless in a
        :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors. See :term:`Glossary <n_jobs>` for more details.
    method : {'lars', 'cd'}, default='lars'
        * `'lars'`: uses the least angle regression method to solve the
          lasso problem (`linear_model.lars_path`);
        * `'cd'`: uses the coordinate descent method to compute the Lasso
          solution (`linear_model.Lasso`). Lars will be faster if the
          estimated components are sparse.
    positive_dict : bool, default=False
        Whether to enforce positivity when finding the dictionary.
    positive_code : bool, default=False
        Whether to enforce positivity when finding the code.
    method_max_iter : int, default=1000
        Maximum number of iterations to perform when solving the lasso
        problem.
    tol : float, default=1e-3
        Control early stopping based on the norm of the differences in the
        dictionary between 2 steps. To disable early stopping based on
        changes in the dictionary, set `tol` to 0.0.
    max_no_improvement : int, default=10
        Control early stopping based on the consecutive number of mini
        batches that does not yield an improvement on the smoothed cost
        function. To disable convergence detection based on cost function,
        set `max_no_improvement` to None.

    Returns
    -------
    code : ndarray of shape (n_samples, n_components), optional
        The sparse code (only returned if `return_code=True`).
    dictionary : ndarray of shape (n_components, n_features), optional
        The solutions to the dictionary learning problem.
    n_iter : int
        Number of iterations run.

    """
    # Check inputs
    X = check_array(X, ensure_min_samples=2)
    if n_components is None:
        n_components = X.shape[1]
    if dict_init is not None:
        dict_init = check_array(dict_init, ensure_min_samples=2)

    # Initialize dictionary and code
    n_samples, n_features = X.shape
    if dict_init is None:
        if positive_dict:
            dict_init = np.abs(np.random.rand(n_components, n_features))
        else:
            dict_init = np.random.rand(n_components, n_features)

    if positive_code:
        code = np.abs(np.random.rand(n_samples, n_components))
    else:
        code = np.random.rand(n_samples, n_components)

    # Initialize variables for tracking
    n_iter = 0
    best_dict = dict_init.copy()
    best_code = code.copy()
    best_loss = np.inf
    last_loss = np.inf
    improvement_window = 0
    no_improvement = 0

    # Shuffle data if necessary
    if shuffle:
        X = X[np.random.permutation(n_samples)]

    # Define stopping criteria
    stop_early = tol > 0.0 or max_no_improvement is not None

    # Define callback function
    if callback is not None:
        def _callback():
            callback(n_iter, best_loss, best_dict, best_code)

    # Iterate over batches
    for batch in range(0, n_samples, batch_size):
        batch_end = min(batch + batch_size, n_samples)

        # Extract batch
        X_batch = X[batch:batch_end]

        # Solve dictionary learning problem
        if method == "lars":
            # Least angle regression (LARS) method
            if positive_dict:
                raise NotImplementedError(
                    "The LARS method does not support positive dictionaries."
                )
            dict_new, code_new, _, _ = lars_path(
                X_batch,
                code,
                n_components=n_components,
                alpha=alpha,
                fit_intercept=False,
                positive=positive_code,
                max_iter=method_max_iter,
            )
        elif method == "cd":
            # Coordinate descent method
            dict_new, code_new, _ = Lasso(
                alpha=alpha,
                fit_intercept=False,
                positive=positive_dict,
                max_iter=method_max_iter,
                tol=tol,
            ).fit(X_batch, code)
        else:
            raise ValueError(f"Unsupported method: {method}")

        # Update dictionary and code
        dict_new = np.copy(dict_new)
        code_new = np.copy(code_new)
        code[batch:batch_end] = code_new
        dict_new = np.copy(dict_new)
        if positive_code:
            code[batch:batch_end] = np.abs(code_new)
        dict_init[batch:batch_end] = dict_new

        # Compute loss
        loss = np.linalg.norm(X - np.dot(code, dict_init)) ** 2 + alpha * np.linalg.