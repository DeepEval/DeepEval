import tensorflow as tf
from typing import Union, Iterable

def check_shapes(func_shapes):
    def decorator(func):
        def wrapper(*args):
            for arg, expected_shape in zip(args, func_shapes):
                if not tf.shape(arg).is_compatible_with(expected_shape):
                    raise ValueError(f"Invalid argument shape: expected {expected_shape}, got {tf.shape(arg)}")
            return func(*args)
        return wrapper
    return decorator

@check_shapes( ("Fmu: [N, Din]", "Fvar: [N, Din]", "Ys.values(): [broadcast N, .]", "return: [broadcast n_funs, N, P]", ) )
def ndiag_mc( funcs: Union[Callable[..., tf.Tensor], Iterable[Callable[..., tf.Tensor]]], S: int, Fmu: TensorType, Fvar: TensorType, logspace: bool = False, epsilon: Optional[TensorType] = None, **Ys: TensorType, ) -> tf.Tensor: """ Computes N Gaussian expectation integrals of one or more functions using Monte Carlo samples. The Gaussians must be independent. `Fmu`, `Fvar`, `Ys` should all have same shape, with overall size `N`. :param funcs: the integrand(s): Callable or Iterable of Callables that operates elementwise :param S: number of Monte Carlo sampling points :param Fmu: array/tensor :param Fvar: array/tensor :param logspace: if True, funcs are the log-integrands and this calculates the log-expectation of exp(funcs) :param Ys: arrays/tensors; deterministic arguments to be passed by name :return: shape is the same as that of the first Fmu """"

    # Your code here
    pass

if __name__ == "__main__":
    # Your runnable example here

    # Sample input values
    n = 100
    s = 500
    d = 5
    fmu = tf.random.normal((n, d))
    fvar = tf.ones((n, d))

    # Example integrand function
    def integrand(x):
        return tf.reduce_sum(x**2)

    # Call the function and print the results
    result = ndiag_mc(integrand, s, fmu, fvar)
    print(result)