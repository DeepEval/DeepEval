import tensorflow as tf
import numpy as np
from typing import Callable, Iterable, Union, Optional
from gpflow.base import TensorType
from gpflow.utilities import to_default_float
from check_shapes import check_shapes

@check_shapes(
    "Fmu: [N, Din]",
    "Fvar: [N, Din]",
    "Ys.values(): [broadcast N, .]",
    "return: [broadcast n_funs, N, P]",
)
def ndiag_mc(
    funcs: Union[Callable[..., tf.Tensor], Iterable[Callable[..., tf.Tensor]]],
    S: int,
    Fmu: TensorType,
    Fvar: TensorType,
    logspace: bool = False,
    epsilon: Optional[TensorType] = None,
    **Ys: TensorType,
) -> tf.Tensor:
    if not isinstance(funcs, Iterable):
        funcs = [funcs]
    
    N, Din = tf.shape(Fmu)[0], tf.shape(Fmu)[1]

    if epsilon is None:
        epsilon = tf.random.normal((S, N, Din), dtype=Fmu.dtype)

    samples = Fmu[None, :, :] + epsilon * tf.sqrt(Fvar[None, :, :])
    
    results = []
    for func in funcs:
        evals = func(samples, **{k: v[None, ...] for k, v in Ys.items()})
        if logspace:
            evals = tf.exp(evals)
        result = tf.reduce_mean(evals, axis=0)
        if logspace:
            result = tf.math.log(result)
        results.append(result)

    return tf.stack(results, axis=0)

if __name__ == "__main__":
    tf.random.set_seed(0)

    # Sample input values
    Fmu = to_default_float(tf.constant([[1.0, 2.0], [3.0, 4.0]]))
    Fvar = to_default_float(tf.constant([[0.1, 0.2], [0.3, 0.4]]))
    S = 1000

    # Define a simple function for testing
    def example_func(x, **Ys):
        return x + Ys['y']

    # Call the function
    result = ndiag_mc(
        funcs=[example_func],
        S=S,
        Fmu=Fmu,
        Fvar=Fvar,
        logspace=False,
        y=tf.constant([[5.0, 6.0], [7.0, 8.0]])
    )

    # Print the results
    print(result)