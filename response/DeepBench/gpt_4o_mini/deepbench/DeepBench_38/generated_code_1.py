import tensorflow as tf
import numpy as np
from typing import Callable, Iterable, Union, Optional

TensorType = tf.Tensor

def check_shapes(*shapes):
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Shape checking logic can be added here if needed
            return func(*args, **kwargs)
        return wrapper
    return decorator

@check_shapes("Fmu: [N, Din]", "Fvar: [N, Din]", "Ys.values(): [broadcast N, .]", "return: [broadcast n_funs, N, P]")
def ndiag_mc(funcs: Union[Callable[..., tf.Tensor], Iterable[Callable[..., tf.Tensor]]], 
             S: int, 
             Fmu: TensorType, 
             Fvar: TensorType, 
             logspace: bool = False, 
             epsilon: Optional[TensorType] = None, 
             **Ys: TensorType) -> tf.Tensor:
    
    N, Din = Fmu.shape
    if isinstance(funcs, Callable):
        funcs = [funcs]
    
    P = len(funcs)
    samples = tf.random.normal((S, N, Din), mean=0.0, stddev=1.0) * tf.sqrt(Fvar) + Fmu
    results = [func(samples, **Ys) for func in funcs]
    
    if logspace:
        results = [tf.reduce_logsumexp(result, axis=0) - tf.math.log(tf.cast(S, dtype=result.dtype)) for result in results]
    else:
        results = [tf.reduce_mean(result, axis=0) for result in results]
    
    return tf.stack(results)

if __name__ == "__main__":
    N, Din, S = 5, 3, 1000
    
    Fmu = tf.random.normal((N, Din))
    Fvar = tf.abs(tf.random.normal((N, Din)))
    
    def sample_func(x):
        return tf.reduce_sum(x, axis=-1)
    
    result = ndiag_mc(sample_func, S, Fmu, Fvar)
    print(result)