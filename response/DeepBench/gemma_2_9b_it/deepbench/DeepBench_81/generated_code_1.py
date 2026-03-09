import tensorflow as tf

def triangular_kernel1d(kernel_size: int) -> tf.Tensor:
  assert kernel_size >= 3
  assert kernel_size % 2 != 0
  kernel_weights = tf.ones(kernel_size, dtype=tf.float32)
  center = kernel_size // 2
  for i in range(center):
    kernel_weights[center-i-1] = kernel_weights[center+i] = i + 1
  return kernel_weights 

if __name__ == "__main__":
  kernel_size = 7
  kernel_weights = triangular_kernel1d(kernel_size)
  print(kernel_weights)