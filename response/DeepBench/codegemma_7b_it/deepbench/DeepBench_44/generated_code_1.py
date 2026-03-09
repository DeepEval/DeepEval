import tensorflow as tf
from tensorflow.keras import backend as K

def gaussian_filter_3d(kernel_sigma):
    if isinstance(kernel_sigma, int):
        kernel_sigma = [kernel_sigma] * 3
    ksize = [3 * k for k in kernel_sigma]
    gaussian_kernel = tf.compat.v1.image.gaussian_filter(tf.zeros(ksize), sigma=kernel_sigma)
    return gaussian_kernel

if __name__ == "__main__":
    sample_input = [[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                   [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
                   [[19, 20, 21], [22, 23, 24], [25, 26, 27]]]
    kernel_sigma = 2
    gaussian_kernel = gaussian_filter_3d(kernel_sigma)
    smoothed_output = K.conv3d(sample_input, gaussian_kernel)
    print(smoothed_output)