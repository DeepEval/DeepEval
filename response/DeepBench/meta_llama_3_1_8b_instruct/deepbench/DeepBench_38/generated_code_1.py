import tensorflow as tf
from tensorflow import TensorType
from typing import Callable, Iterable, Union, Optional
import numpy as np

def ndiag_mc(funcs, S, Fmu, Fvar, logspace=False, epsilon=None, **Ys):
    @tf.function(jit_compile=False)
    def func(x, *args, **kwargs):
        if logspace:
            return tf.reduce_logsumexp(funcs(x, *args, **kwargs), axis=0)
        else:
            return tf.reduce_sum(funcs(x, *args, **kwargs), axis=0)

    Xs = tf.random.normal(shape=(S, Fmu.shape[0]), dtype=Fmu.dtype)
    Ys_tensor = {key: tf.cast(value, dtype=Fmu.dtype) for key, value in Ys.items()}

    for arg in Ys_tensor:
        Ys_tensor[arg] = tf.broadcast_to(Ys_tensor[arg], [S, Fmu.shape[0]])

    return func(Xs, **Ys_tensor)

if __name__ == "__main__":
    # Create sample input values
    funcs = [lambda x: x**2, lambda x: x**3]
    S = 1000
    Fmu = tf.random.normal(shape=(S, 10), dtype=tf.float32)
    Fvar = tf.ones_like(Fmu)
    Ys = {'a': tf.ones_like(Fmu), 'b': tf.random.normal(shape=(S, 10), dtype=tf.float32)}

    # Call the function and print the results
    result = ndiag_mc(funcs, S, Fmu, Fvar)
    print(result.shape)
    print(result)