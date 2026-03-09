import tensorflow as tf

def cauchy_kernel1d(sigma: int) -> tf.Tensor:
  assert sigma > 0
  dim = tf.range(-5, 6)
  return 1 / (tf.math.pi * sigma * (1 + (dim**2) / (sigma**2)))

if __name__ == "__main__":
  sigma = 2
  kernel = cauchy_kernel1d(sigma)
  print(kernel)