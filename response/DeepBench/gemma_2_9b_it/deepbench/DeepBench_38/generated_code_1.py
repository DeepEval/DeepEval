import tensorflow as tf
from typing import Union, Iterable, Optional

def ndiag_mc( funcs: Union[Callable[..., tf.Tensor], Iterable[Callable[..., tf.Tensor]]], S: int, Fmu: tf.TensorType, Fvar: tf.TensorType, logspace: bool = False, epsilon: Optional[tf.TensorType] = None, **Ys: tf.TensorType, ) -> tf.Tensor:
    if isinstance(funcs, callable):
        funcs = [funcs]
    with tf.compat.v1.Session() as sess:
      
        samples = sess.run(tf.random.normal((S,) + Fmu.shape[1:], mean=Fmu, stddev=tf.sqrt(Fvar)))
        if logspace:
            return tf.reduce_mean(tf.exp(tf.reduce_sum( [func( *samples, **Ys) for func in funcs], axis=-1)), axis=0)
        else:
            return tf.reduce_mean(tf.stack([func(*samples, **Ys) for func in funcs], axis=0), axis=0)
if __name__ == "__main__":
    Fmu = tf.constant([0.0, 1.0])
    Fvar = tf.constant([0.1, 0.2])
    Ys = {'x': tf.constant([2.0, 3.0])}
    
    def f(x, y):
      return x**2 + y

    result = ndiag_mc(f, S=1000, Fmu=Fmu, Fvar=Fvar, **Ys)
    print(result)