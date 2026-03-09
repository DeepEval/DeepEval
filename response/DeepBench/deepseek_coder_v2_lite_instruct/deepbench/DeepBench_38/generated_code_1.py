import tensorflow as tf
from typing import Union, Callable, Iterable, Optional

def ndiag_mc(funcs: Union[Callable[..., tf.Tensor], Iterable[Callable[..., tf.Tensor]]], S: int, Fmu: tf.Tensor, Fvar: tf.Tensor, logspace: bool = False, epsilon: Optional[tf.Tensor] = None, **Ys: tf.Tensor) -> tf.Tensor:
    # Import necessary functions
    from tensorflow.random import normal
    
    # Check shapes
    assert Fmu.shape == Fvar.shape, "Fmu and Fvar must have the same shape"
    N, Din = Fmu.shape
    for key, value in Ys.items():
        assert value.shape == (N, *value.shape[1:]), f"{key} must have shape [N, ...]"
    
    # Sample from standard normal
    if epsilon is None:
        epsilon = normal([S, N, Din])
    else:
        assert epsilon.shape == (S, N, Din), "epsilon must have shape [S, N, Din]"
    
    # Expand dimensions for broadcasting
    Fmu = tf.expand_dims(Fmu, axis=0)  # [1, N, Din]
    Fvar = tf.expand_dims(Fvar, axis=0)  # [1, N, Din]
    epsilon = tf.expand_dims(epsilon, axis=2)  # [S, N, 1, Din]
    
    # Compute the samples
    X = Fmu + tf.sqrt(Fvar) * epsilon
    
    # Evaluate the functions
    if callable(funcs):
        funcs = [funcs]
    results = []
    for func in funcs:
        func_results = func(X, **Ys)  # [S, N, P]
        if logspace:
            func_results = tf.math.exp(func_results)
        results.append(tf.reduce_mean(func_results, axis=0))  # [N, P]
    
    # Stack results
    return tf.stack(results, axis=0)  # [n_funs, N, P]

if __name__ == "__main__":
    # Define a simple integrand
    def integrand(X):
        return tf.reduce_sum(X**2, axis=-1)
    
    # Create sample inputs
    N = 3
    Din = 2
    S = 1000
    Fmu = tf.ones([N, Din])
    Fvar = tf.ones([N, Din])
    Ys = {'Y': tf.ones([N, 2])}
    
    # Call the function
    result = ndiag_mc(integrand, S, Fmu, Fvar, logspace=False, **Ys)
    
    # Print the results
    print(result)