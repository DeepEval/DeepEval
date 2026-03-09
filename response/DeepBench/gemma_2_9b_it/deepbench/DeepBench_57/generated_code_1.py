import tensorflow as tf
from typing import Callable, Optional, Tuple

def mvnquad(func: Callable[[tf.Tensor], tf.Tensor], means: tf.Tensor, covs: tf.Tensor, H: int, Din: Optional[int] = None, Dout: Optional[Tuple[int, ...]] = None, ) -> tf.Tensor:
  if Din is None:
    Din = means.shape[1]
  if Din is None:
    raise ValueError( "If `Din` is passed as `None`, `means` must have a known shape. " "Running mvnquad in `autoflow` without specifying `Din` and `Dout` " "is problematic. Consider using your own session." )
  if Dout is None:
    Dout = ()
  
  z = tf.random.normal([H, Din])
  weights = tf.math.exp(-0.5 * tf.reduce_sum(z**2, axis=1))
  
  input_to_func = means + tf.linalg.sqrtm(covs) @ z
  output = func(input_to_func)
  return tf.reduce_sum(weights[:,None] * output, axis=0) 

if __name__ == "__main__":
  means = tf.constant([[1.0, 2.0]])
  covs = tf.constant([[[0.1, 0.0], [0.0, 0.2]]])
  H = 10
  Din = 2
  Dout = 1

  def example_function(x):
    return tf.square(x[:,0]) + tf.square(x[:,1])

  result = mvnquad(example_function, means, covs, H, Din, Dout)
  print(result)