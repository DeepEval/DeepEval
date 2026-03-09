import tensorflow as tf

def gaussian_filter_3d(kernel_sigma):
  if isinstance(kernel_sigma, int):
    kernel_sigma = [kernel_sigma, kernel_sigma, kernel_sigma]
  
  sigmas = tf.cast(kernel_sigma, dtype=tf.float32)
  
  def gaussian(x, sigma):
    return tf.exp(-0.5 * tf.reduce_sum(tf.square(x) / tf.square(sigma), axis=-1))

  
  ranges = [tf.range(-k, k + 1) for k in kernel_sigma]
  x, y, z = tf.meshgrid(*ranges, indexing='ij')
  
  kernel = gaussian(tf.stack((x, y, z), axis=-1), sigmas)
  
  return kernel / tf.reduce_sum(kernel)

if __name__ == "__main__":
  kernel_sigma = 2
  kernel = gaussian_filter_3d(kernel_sigma)
  print(kernel.shape)
  print(kernel)